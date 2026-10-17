"""Run beef horizon and ablation experiments; outputs go to results/beef_experiments.

Run: python beef_experiments.py
Prepare data and check folds only: python beef_experiments.py --prepare-only
"""

import argparse
import json
import warnings
import time
from uuid import uuid4
from pathlib import Path
from joblib import parallel_config
from threadpoolctl import threadpool_limits

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/beef_experiments"
SEED = 2210
N_INNER = 3
N_JOBS = 2
TARGETS = ["live weight", "carcass weight", "price", "Total number of defects"]
ANIMALS = ["steer", "heifer"]
WEATHER = ["MnDir(degT)", "MnSpd(mps)", "MnTemp(C)", "MnRH(%)", "Rain(mm)",
           "RadGlb(MJ/m2)", "Average W/m2", "Predicted BGT", "THI", "HLI", "Grazing HLI2"]
STATIC = ["NZDI", "Transport distance (calculated)", "lairage time (calculated)", "Age"]


class ConvergedPipeline(Pipeline):
    # run inside each worker: non-convergence invalidates this candidate's CV score
    def fit(self, X, y=None, **params):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return super().fit(X, y, **params)


def model_specs(classification):
    # add future models here as name: (estimator, parameter grid)
    # names identify saved results; use a new name when changing a model's search
    if classification:
        specs = {"LogisticRegression": (
            LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                               max_iter=20000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10], "model__l1_ratio": [0.1, 0.5, 0.9]})}
    else:
        specs = {"ElasticNet": (
            ElasticNet(max_iter=50000, random_state=SEED),
            {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000], "model__l1_ratio": [0.1, 0.5, 0.9]})}

    # fixed, compact grids are reused at every horizon and outer fold
    svm = SVC(class_weight="balanced", max_iter=100000, random_state=SEED) if classification else SVR(max_iter=100000)
    svm_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.001, 0.01]}
    if not classification:
        svm_grid["model__epsilon"] = [0.1, 1]
    forest = (RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=SEED)
              if classification else RandomForestRegressor(n_jobs=1, random_state=SEED))
    lightgbm = (LGBMClassifier(class_weight="balanced", n_jobs=1, verbosity=-1, random_state=SEED)
                if classification else LGBMRegressor(n_jobs=1, verbosity=-1, random_state=SEED))
    # disable automatic random validation splits: early stopping would ignore date groups
    hist = (HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=SEED)
            if classification else HistGradientBoostingRegressor(early_stopping=False, random_state=SEED))
    specs.update({
        "SVC" if classification else "SVR": (svm, svm_grid),
        "RandomForest": (forest, {"model__n_estimators": [300], "model__max_depth": [None, 6],
                                  "model__min_samples_leaf": [1, 5, 15], "model__max_features": [0.5, 1.0]}),
        "LightGBM": (lightgbm, {"model__n_estimators": [150, 300], "model__learning_rate": [0.03, 0.1],
                               "model__num_leaves": [7, 15], "model__min_child_samples": [10, 30]}),
        "HistGradientBoosting": (hist, {"model__max_iter": [150, 300], "model__learning_rate": [0.03, 0.1],
                                       "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0, 1]})
    })
    return specs


def prepare_data():
    beef = pd.read_csv(ROOT / "data/beef_preprocessed.csv")
    old = pd.read_csv(ROOT / "data/beef_weather_windows.csv")
    weather = pd.read_csv(ROOT / "data/weather_preprocessed.csv", low_memory=False)
    beef["kill date"] = pd.to_datetime(beef["kill date"])
    weather["date"] = pd.to_datetime(weather["Date(NZST)"])

    # preserve the positional animal/window join before filtering by estimated age
    # four months means 120 days using the preparation notebook's 30-day conversion
    data = beef.join(old)
    data = data.loc[data["Animal Type"].isin(ANIMALS) & data["Age"].ge(4)].copy()
    data["animal_index"] = data.index
    # retain the original whole-series thresholds for comparability with the 60-day features
    thresholds = weather[WEATHER[1:]].quantile(0.9)
    additions = []
    for date in tqdm(data["kill date"].sort_values().unique(), desc="Weather windows"):
        feats = {"kill date": date}
        for lo, hi in [(61, 90), (91, 120)]:
            sub = weather.loc[weather["date"].between(
                date - pd.Timedelta(days=hi), date - pd.Timedelta(days=lo))]
            for col in WEATHER:
                values = sub[col].dropna()
                prefix = f"{col}__win{lo}_{hi}_"
                stats = {"max": values.max(), "min": values.min(),
                         "diff": values.max() - values.min(), "mean": values.mean(),
                         "std": values.std(ddof=0)}
                feats.update({prefix + k: v for k, v in stats.items()})
                if col in thresholds:
                    thr = thresholds[col]
                    feats[prefix + f"above{round(thr, 2)}_count"] = (
                        (values > thr).sum() if len(values) else np.nan)
        additions.append(feats)
    extra = pd.DataFrame(additions).set_index("kill date")
    data = data.join(extra, on="kill date")
    base = [c for c in old if "__win" in c]
    horizons = {60: base, 90: base + [c for c in extra if "__win61_90_" in c],
                120: base + list(extra.columns)}
    # NZDI is environmental: exclude it from the non-weather comparator
    settings = {"Non-weather only": [c for c in STATIC if c != "NZDI"]}
    # the non-weather setting is shared; the other settings add successively older exposure
    for horizon, cols in horizons.items():
        settings[f"Weather only {horizon}"] = cols
        settings[f"Non-weather + weather {horizon}"] = STATIC + cols
    return data, settings


def assign_folds(data):
    # save once and reuse across targets, horizons and subsequent runs
    path = OUT / "fold_mapping.csv"
    if path.exists():
        mapping = pd.read_csv(path, parse_dates=["kill date"])
    else:
        rows = []
        for animal in ANIMALS:
            counts = (data.loc[data["Animal Type"].eq(animal)]
                      .groupby("kill date").size().sort_values(ascending=False))
            sizes = np.zeros(10, dtype=int)
            for date, count in counts.items():
                fold = sizes.argmin()
                rows.append({"Animal Type": animal, "kill date": date, "fold": fold + 1})
                sizes[fold] += count
        mapping = pd.DataFrame(rows)
        mapping.to_csv(path, index=False)
    return data.merge(mapping, on=["Animal Type", "kill date"], how="left")


def make_splits(subset, classification):
    y = subset["Total number of defects"].gt(0)
    splits = []
    for fold in range(1, 11):
        train = np.flatnonzero(subset["fold"].ne(fold))
        test = np.flatnonzero(subset["fold"].eq(fold))
        inner = list(GroupKFold(N_INNER, shuffle=True, random_state=SEED).split(
            subset.iloc[train], groups=subset["kill date"].iloc[train]))
        # class checks protect balanced accuracy, rather than validate file formats
        if classification and (y.iloc[test].nunique() < 2 or any(
                y.iloc[train].iloc[idx].nunique() < 2 for pair in inner for idx in pair)):
            raise ValueError(f"Single-class split in outer fold {fold}")
        splits.append((fold, train, test, inner))
    return splits


def describe_folds(data):
    # appendix counts use each target's available outcomes, before any model is fitted
    counts = data.groupby(["Animal Type", "fold"]).agg(
        animals=("animal_index", "size"), slaughter_dates=("kill date", "nunique"))
    counts.to_csv(OUT / "fold_sizes.csv")
    rows = []
    for animal in ANIMALS:
        for target in TARGETS:
            sub = data.loc[data["Animal Type"].eq(animal)].dropna(subset=[target])
            make_splits(sub, target == "Total number of defects")
            for fold, group in sub.groupby("fold"):
                rows.append({"animal_type": animal, "target": target, "fold": fold,
                             "animals": len(group), "slaughter_dates": group["kill date"].nunique(),
                             "positive": group[target].gt(0).sum() if target == TARGETS[-1] else np.nan,
                             "negative": group[target].eq(0).sum() if target == TARGETS[-1] else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "fold_sizes_by_target.csv", index=False)


RESULT_KEY = ["animal_type", "target", "setting", "model", "fold"]
LOCKED_EXPORTS = set()


def latest_csv(name):
    # a locked export may be older than its durable fallback checkpoint
    files = list((OUT / "checkpoints").glob(f"{Path(name).stem}-*.csv"))
    if (OUT / name).exists():
        files.append(OUT / name)
    return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None


def save_csv(rows, name):
    # replace each file atomically so interruption cannot leave a partly written CSV
    path = OUT / name
    folder = OUT / "checkpoints"
    folder.mkdir(exist_ok=True)
    temporary = folder / f"{path.stem}-{uuid4().hex}.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    for delay in [0, 0.2, 0.5, 1]:
        time.sleep(delay)
        try:
            temporary.replace(path)
            return
        except PermissionError:
            pass
    # Windows readers can lock an existing CSV against replacement. Keep the new
    # snapshot under a unique name; resume reads it automatically, so no fit is lost.
    checkpoint = folder / f"{path.stem}-{time.time_ns()}-{uuid4().hex}.csv"
    temporary.replace(checkpoint)
    if name not in LOCKED_EXPORTS:
        tqdm.write(f"{name} is locked; saved a fallback checkpoint. Experiments will continue.")
        LOCKED_EXPORTS.add(name)


def run_experiments(data, settings):
    def read_rows(name):
        path = latest_csv(name)
        return pd.read_csv(path).to_dict("records") if path is not None else []

    def key(row):
        return tuple(row[k] for k in RESULT_KEY)

    # a score row marks completion; predictions are saved first below
    scores = read_rows("fold_scores.csv")
    # earlier runs allowed unconverged candidates to compete; retain a backup and redo them
    if any(row.get("convergence_policy") != "reject" and row["model"] != "Dummy" for row in scores):
        for filename in ["fold_scores.csv", "predictions.csv"]:
            backup = OUT / filename.replace(".csv", "_before_convergence_rejection.csv")
            if not backup.exists():
                backup.write_bytes((OUT / filename).read_bytes())
        scores = [row for row in scores if row.get("convergence_policy") == "reject" or row["model"] == "Dummy"]
    completed = {key(row) for row in scores}
    failures = {key(row): row for row in read_rows("failed_fits.csv")}

    def save_failures():
        columns = RESULT_KEY + ["stage", "error", "best_params", "rejected_candidates"]
        save_csv(pd.DataFrame(failures.values(), columns=columns), "failed_fits.csv")
    predictions = [row for row in read_rows("predictions.csv") if key(row) in completed]
    total = sum(10 * (len(settings) * len(model_specs(target == TARGETS[-1])) + 1)
                for animal in ANIMALS for target in TARGETS)
    with tqdm(total=total, desc="Outer fits") as progress:
        for animal in ANIMALS:
            for target in TARGETS:
                sub = data.loc[data["Animal Type"].eq(animal)].dropna(subset=[target])
                classification = target == TARGETS[-1]
                y = sub[target].gt(0).astype(int) if classification else sub[target]
                if classification:
                    scoring, metric, evaluate = "balanced_accuracy", "Balanced accuracy", balanced_accuracy_score
                    dummy = DummyClassifier(strategy="most_frequent")
                else:
                    scoring, metric, evaluate = "neg_mean_absolute_error", "MAE", mean_absolute_error
                    dummy = DummyRegressor(strategy="mean")
                # each model gets the same data, settings and inner/outer date splits
                jobs = [("Dummy", dummy, {}, "Dummy", STATIC)]
                jobs += [(name, model, grid, setting, columns)
                         for name, (model, grid) in model_specs(classification).items()
                         for setting, columns in settings.items()]
                for fold, train, test, inner in make_splits(sub, classification):
                    for name, model, grid, setting, columns in jobs:
                        info = {"animal_type": animal, "target": target, "setting": setting, "model": name}
                        run_key = key({**info, "fold": fold})
                        if run_key in completed:
                            progress.update()
                            continue
                        progress.set_postfix_str(f"{animal}, {target}, {name}, fold {fold}, {setting}")
                        X = sub[columns]
                        pipeline = ConvergedPipeline([
                            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scaler", StandardScaler()), ("model", clone(model))])
                        # search without automatic refitting, so tuning and final-fit failures are distinct
                        stage, best_params, rejected = "tuning", {}, 0
                        try:
                            if name == "Dummy":
                                fitted = clone(model)
                            else:
                                search = GridSearchCV(pipeline, grid, scoring=scoring, cv=inner,
                                                      n_jobs=N_JOBS, error_score=np.nan, refit=False)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", FitFailedWarning)
                                    warnings.filterwarnings("ignore", message="One or more of the test scores.*", category=UserWarning)
                                    # parallelise candidates without oversubscribing CPU threads
                                    with parallel_config(backend="loky", inner_max_num_threads=1):
                                        search.fit(X.iloc[train], y.iloc[train])
                                means = search.cv_results_["mean_test_score"]
                                valid = np.isfinite(means)
                                rejected = int((~valid).sum())
                                if not valid.any():
                                    raise ValueError("No candidate converged in every inner fold")
                                best = np.argmax(np.where(valid, means, -np.inf))
                                best_params = search.cv_results_["params"][best]
                                fitted = clone(pipeline).set_params(**best_params)
                            stage = "refit"
                            with threadpool_limits(limits=1):
                                fitted.fit(X.iloc[train], y.iloc[train])
                            stage = "prediction"
                            pred = fitted.predict(X.iloc[test])
                        except Exception as error:
                            # keep unsuccessful jobs pending and continue the unattended run
                            failures[run_key] = {**info, "fold": fold, "stage": stage,
                                                 "error": f"{type(error).__name__}: {error}",
                                                 "best_params": json.dumps(best_params),
                                                 "rejected_candidates": rejected}
                            save_failures()
                            progress.update()
                            continue
                        scores.append({**info, "fold": fold, "metric": metric,
                                       "convergence_policy": "reject", "rejected_candidates": rejected,
                                       "score": evaluate(y.iloc[test], pred), "n_train": len(train),
                                       "n_test": len(test), "n_features": len(columns),
                                       "n_train_dates": sub["kill date"].iloc[train].nunique(),
                                       "n_test_dates": sub["kill date"].iloc[test].nunique(),
                                       "best_params": json.dumps(best_params)})
                        predictions.extend({**info, "fold": fold, "animal_index": idx,
                                            "kill_date": date, "observed": obs, "predicted": fitted}
                                           for idx, date, obs, fitted in zip(sub["animal_index"].iloc[test],
                                               sub["kill date"].iloc[test], y.iloc[test], pred))
                        # save each fit; adding models or restarting skips completed combinations
                        save_csv(predictions, "predictions.csv")
                        save_csv(scores, "fold_scores.csv")
                        failures.pop(run_key, None)
                        save_failures()
                        completed.add(run_key)
                        progress.update()
    pooled = []
    # headline metrics pool held-out animals, rather than averaging unequal folds
    keys = ["animal_type", "target", "setting", "model"]
    for key, group in pd.DataFrame(predictions, columns=keys + ["fold", "observed", "predicted"]).groupby(keys):
        if group["fold"].nunique() != 10:
            continue
        classification = key[1] == TARGETS[-1]
        evaluate = balanced_accuracy_score if classification else mean_absolute_error
        pooled.append({**dict(zip(keys, key)), "n": len(group),
                       "metric": "Balanced accuracy" if classification else "MAE",
                       "score": evaluate(group["observed"], group["predicted"])})
    save_csv(pd.DataFrame(pooled, columns=keys + ["n", "metric", "score"]), "pooled_scores.csv")
    # refresh exports even when every fit was already complete at restart
    save_csv(predictions, "predictions.csv")
    save_csv(scores, "fold_scores.csv")
    save_failures()
    print(f"Finished: {len(failures)} fits pending.")
    if LOCKED_EXPORTS:
        print("Some exports were locked. Close programs holding the CSVs and rerun to refresh them; "
              "completed fits will be loaded from checkpoints and skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, settings = prepare_data()
    data = assign_folds(data)
    describe_folds(data)
    if not args.prepare_only:
        run_experiments(data, settings)

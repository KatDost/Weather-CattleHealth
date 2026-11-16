"""Run beef direction experiments with nested date-grouped CV.

Run: python beef_direction_experiments_10fold.py
Prepare data and check folds only: python beef_direction_experiments_10fold.py --prepare-only
Verify completed results: python beef_direction_experiments_10fold.py --verify-only
"""
import argparse
import hashlib
import json
import os
import re
import warnings
import time
from pathlib import Path
from uuid import uuid4

# This runner is single-worker; avoid joblib's obsolete WMIC probe on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import sklearn
import lightgbm
import scipy
import platform
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/beef_direction_comparison_10fold"
KEY = ["kill date", "Teeth", "Animal Type"]
MATCH = KEY[1:]
TARGETS = ["live weight", "carcass weight", "price"]
ANIMALS = ["heifer", "steer"]
SETTINGS = ["controls", "nzdi", "weather_pruned"]
CORRELATION_LIMIT = 0.95
SEED = 2210
N_OUTER = 10
N_INNER = 3
HORIZONS = [30, 60, 120, 180]
FIT_KEY = ["animal_type", "target", "model", "setting", "horizon", "fold"]
CELL_KEY = FIT_KEY[:-1]
LOCKED_EXPORTS = set()



def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_csv(rows, name, out=None):
    # Match the beef runner's atomic exports and Windows-reader fallback.
    out = OUT if out is None else out
    folder = out / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{Path(name).stem}-{uuid4().hex}.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    try:
        temporary.replace(out / name)
    except PermissionError:
        temporary.replace(folder / f"{Path(name).stem}-{time.time_ns()}-{uuid4().hex}.csv")
        if name not in LOCKED_EXPORTS:
            tqdm.write(f"{name} is locked; saved a fallback checkpoint. Experiments will continue.")
            LOCKED_EXPORTS.add(name)


def load_data(out):
    """Snapshot once. Never write to source data or regression outputs."""
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_path = inputs / "manifest.json"
    names = ["beef_preprocessed.csv", "beef_weather_windows.csv"]
    if not manifest_path.exists():
        hashes = {}
        for name in names:
            source = ROOT / "data" / name
            content = source.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            (inputs / name).write_bytes(content)
            if digest(source) != hashes[name]:
                raise RuntimeError("Input changed during snapshot; retry preparation.")
        save_json(manifest_path, hashes)
    hashes = json.loads(manifest_path.read_text())
    if any(digest(inputs / name) != hashes[name] for name in names):
        raise ValueError("Snapshot changed; use a new experiment directory/version.")
    beef = pd.read_csv(inputs / names[0], parse_dates=["kill date"], float_precision="round_trip")
    windows = pd.read_csv(inputs / names[1], float_precision="round_trip")
    if len(beef) != len(windows):
        raise ValueError("Animal/weather row counts differ.")
    if beef["kill date"].isna().any():
        raise ValueError("Missing slaughter date.")
    # Include the user's full-period exceedance counts alongside all window summaries.
    weather = [c for c in windows if re.search(r'__win\d+_(\d+)_', c) and
               int(re.search(r'__win\d+_(\d+)_', c)[1]) <= max(HORIZONS)]
    # Necessary alignment check; it cannot prove identity within a date.
    if windows[weather].groupby(beef["kill date"]).nunique(dropna=False).gt(1).any().any():
        raise ValueError("Weather rows disagree within slaughter date; check alignment.")
    data = beef.join(windows[weather])
    data = data[data["Animal Type"].isin(["heifer", "steer"])].copy()
    exclusions = data.groupby("Animal Type")["Teeth"].agg(["size", "count"])
    exclusions["missing_teeth"] = exclusions["size"] - exclusions["count"]
    exclusions.to_csv(out / "animal_eligibility.csv")
    data = data.dropna(subset=["Teeth"])
    if not data["Teeth"].isin(range(9)).all():
        raise ValueError("Unexpected recorded teeth values.")
    return data, weather


def group_data(data, target, weather):
    usable = data.dropna(subset=[target])
    groups = usable.groupby(KEY, sort=True)
    result = groups[target].agg(["mean", "size"]).rename(columns={"mean": "outcome", "size": "animals"})
    result["values"] = groups[target].agg(list)
    result = result.join(groups[["NZDI"] + weather].mean()).reset_index()
    result["category"] = result["Teeth"].astype(int).astype(str) + "/" + result["Animal Type"]
    return result


def labels(train, held=None):
    """Training means leave their own date out; held-out means use training only."""
    reference = train.groupby(MATCH)["outcome"].agg(["sum", "count"])
    frame = (train if held is None else held).copy().join(reference, on=MATCH)
    frame["reference_dates"] = frame["count"] - (1 if held is None else 0)
    frame["reference"] = (frame["sum"] - (frame["outcome"] if held is None else 0)) / frame["reference_dates"]
    frame["exclusion"] = np.where(frame["reference_dates"].fillna(0).le(0), "no_reference",
                                 np.where(frame["outcome"].eq(frame["reference"]), "tie", ""))
    frame["label"] = frame["outcome"].gt(frame["reference"]).astype(int)
    return frame.drop(columns=["sum", "count"])


def eligible(frame):
    return frame.loc[frame["exclusion"].eq("")].copy()


def weights(frame):
    return 1 / frame.groupby("kill date")["kill date"].transform("size").to_numpy()


def splits(frame, n):
    dates = np.sort(frame["kill date"].unique())
    for a, b in KFold(n, shuffle=True, random_state=SEED).split(dates):
        yield frame[frame["kill date"].isin(dates[a])], frame[frame["kill date"].isin(dates[b])]


def metrics(frame, prob):
    w = weights(frame)
    y = frame["label"].to_numpy()
    pair = float(w[y == 0].sum() * w[y == 1].sum())
    return {"auc": float(roc_auc_score(y, prob, sample_weight=w)) if pair else None,
            "pair_weight": pair,
            "brier_score": float(brier_score_loss(y, prob, sample_weight=w)),
            "log_loss": float(log_loss(y, prob, sample_weight=w, labels=[0, 1]))}


def model_specs():
    # Model and horizon are explicit experiment factors, never search parameters.
    return {
        "ElasticNet": (LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                           solver="saga", max_iter=20000, random_state=SEED),
                       [{"C": 0.1}, {"C": 1.0}]),
        "RandomForest": (RandomForestClassifier(n_estimators=300, max_features=0.5,
                             n_jobs=1, random_state=SEED),
                         [{"min_samples_leaf": 2}, {"min_samples_leaf": 5}]),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=300, max_features=0.5,
                           n_jobs=1, random_state=SEED),
                       [{"min_samples_leaf": 2}, {"min_samples_leaf": 5}]),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=150,
                                     learning_rate=0.05, min_samples_leaf=5,
                                     l2_regularization=1, early_stopping=False, random_state=SEED),
                                 [{"max_leaf_nodes": 3}, {"max_leaf_nodes": 7}]),
        "LightGBM": (LGBMClassifier(n_estimators=150, learning_rate=0.05,
                         min_child_samples=5, reg_lambda=1, n_jobs=1,
                         verbosity=-1, deterministic=True, force_col_wise=True,
                         random_state=SEED),
                     [{"num_leaves": 3}, {"num_leaves": 7}]),
    }


def jobs():
    yield "Dummy", "dummy", 0
    for name in model_specs():
        yield name, "controls", 0
        yield name, "nzdi", 0
        for horizon in HORIZONS:
            yield name, "weather_pruned", horizon


def select_weather(train, weather, horizon):
    # Apply the fixed horizon BEFORE learning training-date correlations.
    columns = [col for col in weather
               if int(re.search(r"__win\d+_(\d+)_", col)[1]) <= horizon]
    dates = train.groupby("kill date")[columns].mean()
    usable = [col for col in columns if dates[col].nunique() > 1]
    observed = dates[usable].notna().sum()
    dates = dates[usable].fillna(dates[usable].median())
    correlation = dates.corr(method="spearman").abs()
    ordered = sorted(usable, key=lambda col: (-observed[col], col))
    retained = []
    for col in ordered:
        if not retained or not correlation.loc[col, retained].ge(CORRELATION_LIMIT).any():
            retained.append(col)
    return retained


def fit_predict(train, test, weather, name, setting, horizon, params):
    if train.empty or test.empty:
        raise ValueError("No eligible training or held-out groups.")
    if train["Animal Type"].nunique() != 1 or set(test["Animal Type"]) != set(train["Animal Type"]):
        raise ValueError("Each fit must contain exactly one matching animal type.")
    w = weights(train)
    if name == "Dummy":
        return np.full(len(test), np.average(train["label"], weights=w)), dict(
            selected_weather=[], feature_names=[], coefficients=[], nonzero_coefficients=None)
    if train["label"].nunique() != 2:
        raise ValueError("Single-class training labels.")
    selected = select_weather(train, weather, horizon) if setting == "weather_pruned" else []
    numeric = ([] if setting == "controls" else ["NZDI"]) + selected
    transformers = [("teeth", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"])]
    if numeric:
        transformers.append(("numeric", make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()), numeric))
    estimator = clone(model_specs()[name][0]).set_params(**params)
    model = Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train, train["label"], model__sample_weight=w)
    probability = model.predict_proba(test)[:, list(model[-1].classes_).index(1)]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probabilities.")
    names = model[0].get_feature_names_out().tolist()
    coefficients = []
    nonzero = None
    if hasattr(model[-1], "coef_"):
        values = model[-1].coef_[0]
        nonzero = int(np.count_nonzero(np.abs(values) > 1e-10))
        coefficients = [dict(feature=feature, coefficient=float(value)) for feature, value in zip(names, values)]
    return probability, dict(selected_weather=selected, feature_names=names,
                             coefficients=coefficients, nonzero_coefficients=nonzero)


def significance(train, held, rng, draws=5000):
    refs = train.groupby(MATCH)["outcome"].agg(list)
    result = []
    for _, row in held.iterrows():
        means = np.asarray(refs.get((row["Teeth"], row["Animal Type"]), []))
        animals = np.asarray(row["values"])
        if len(means) < 2 or len(animals) < 2:
            result.append((None, None, None))
            continue
        diff = rng.choice(animals, (draws, len(animals))).mean(axis=1) - rng.choice(means, (draws, len(means))).mean(axis=1)
        lo, hi = np.quantile(diff, [.025, .975])
        result.append((float(lo), float(hi), bool(lo > 0 or hi < 0)))
    return result


def count_row(frame, target, outer, inner, role):
    valid = eligible(frame)
    return dict(target=target, outer=outer, inner=inner, role=role, groups=len(frame),
                dates=int(frame["kill date"].nunique()), eligible_groups=len(valid),
                eligible_dates=int(valid["kill date"].nunique()),
                positive=int(valid["label"].sum()), negative=int(len(valid)-valid["label"].sum()),
                ties=int(frame["exclusion"].eq("tie").sum()),
                no_reference=int(frame["exclusion"].eq("no_reference").sum()))


def tune(train, weather, name, setting, horizon):
    if name == "Dummy":
        return {}, []
    diagnostics, choices = [], []
    for index, params in enumerate(tqdm(model_specs()[name][1], desc="Inner candidates", leave=False)):
        rows = []
        for inner, (a, b) in enumerate(splits(train, N_INNER), 1):
            row = dict(candidate=index, params=params, inner=inner)
            a, b = eligible(labels(a)), eligible(labels(a, b))
            row.update(train_groups=len(a), validation_groups=len(b),
                       train_dates=int(a["kill date"].nunique()), validation_dates=int(b["kill date"].nunique()))
            try:
                probability, details = fit_predict(a, b, weather, name, setting, horizon, params)
                row.update(metrics(b, probability), n_features=len(details["feature_names"]),
                           selected_weather=details["selected_weather"],
                           nonzero_coefficients=details["nonzero_coefficients"],
                           probability_min=float(probability.min()), probability_max=float(probability.max()))
            except Exception as error:
                row["failure"] = repr(error)
            rows.append(row)
            diagnostics.append(row)
        if all("failure" not in row for row in rows):
            pair = sum(row["pair_weight"] for row in rows)
            if pair:
                auc = sum((row["auc"] or 0) * row["pair_weight"] for row in rows) / pair
                loss = np.average([row["log_loss"] for row in rows],
                                  weights=[row["validation_dates"] for row in rows])
                choices.append((auc, -float(loss), -index, params))
    # Original candidate order breaks exact ties; neither model nor horizon changes.
    return (max(choices, key=lambda value: value[:3])[3] if choices else None), diagnostics


def configuration(weather):
    return dict(pipeline_version=4, seed=SEED, outer=N_OUTER, inner=N_INNER,
                exceedance_thresholds="precomputed over the full weather period; included",
                animals=ANIMALS, targets=TARGETS, settings=SETTINGS, horizons=HORIZONS,
                correlation_limit=CORRELATION_LIMIT, bootstrap_draws=5000,
                tuning="within-fold pair-weighted AUC; date-weighted log loss; candidate order",
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              sklearn=sklearn.__version__, scipy=scipy.__version__, lightgbm=lightgbm.__version__),
                models={name: dict(estimator=repr(model), grid=grid) for name, (model, grid) in model_specs().items()},
                weather_columns=weather, input_hashes=json.loads((OUT / "inputs/manifest.json").read_text()),
                source_sha256=digest(Path(__file__)))


def prepare_data():
    OUT.mkdir(parents=True, exist_ok=True)
    data, weather = load_data(OUT)
    config = configuration(weather)
    path = OUT / "run_manifest.json"
    source = OUT / "beef_direction_experiments_10fold.py"
    if path.exists():
        saved = json.loads(path.read_text())
        archived_hash = saved["source_sha256"]
        runner_hash = saved.pop("runner_sha256", archived_hash)
        if digest(source) != archived_hash or dict(saved, source_sha256=runner_hash) != config:
            raise ValueError("Code, dependencies, inputs or design changed. Use a fresh OUT directory.")
    else:
        save_json(path, config)
    if not source.exists():
        source.write_bytes(Path(__file__).read_bytes())
    return data, weather


def describe_folds(data, weather):
    audit, membership, plans = [], [], []
    for animal in ANIMALS:
        for target in TARGETS:
            groups = group_data(data[data["Animal Type"].eq(animal)], target, weather)
            save_csv(groups.drop(columns=["values"]), f"{animal}_{target.replace(' ', '_')}_groups.csv")
            for fold, (train, test) in enumerate(splits(groups, N_OUTER), 1):
                a, b = labels(train), labels(train, test)
                for role, frame in [("train", a), ("test", b)]:
                    audit.append(dict(animal_type=animal, **count_row(frame, target, fold, None, role)))
                    membership.extend(dict(animal_type=animal, target=target, outer=fold, inner=None,
                                           role=role, kill_date=str(date)) for date in frame["kill date"].unique())
                for inner, (it, iv) in enumerate(splits(train, N_INNER), 1):
                    for role, frame in [("train", labels(it)), ("validation", labels(it, iv))]:
                        audit.append(dict(animal_type=animal, **count_row(frame, target, fold, inner, role)))
                        membership.extend(dict(animal_type=animal, target=target, outer=fold, inner=inner,
                                               role=role, kill_date=str(date)) for date in frame["kill date"].unique())
                prefix = f"{animal}_{target.replace(' ', '_')}_fold{fold}"
                save_csv(a.drop(columns=["values"]), prefix + "_train_labels.csv")
                save_csv(b.drop(columns=["values"]), prefix + "_test_labels.csv")
                plans.append((animal, target, fold, train, eligible(a), eligible(b)))
    save_csv(audit, "fold_counts.csv")
    save_csv(membership, "fold_mapping.csv")
    save_csv([dict(zip(FIT_KEY, [animal, target, name, setting, horizon, fold]))
              for animal, target, fold, *_ in plans for name, setting, horizon in jobs()], "expected_fits.csv")
    return plans


def fit_path(info):
    return OUT / "fits" / ("__".join(str(info[key]).replace(" ", "_") for key in FIT_KEY) + ".json")


def records_for_fit(info, held, probability, intervals):
    rows = []
    for (_, row), prob, (lo, hi, sig) in zip(held.iterrows(), probability, intervals):
        rows.append(dict(**info, **{k: str(row[k]) if k == "kill date" else row[k] for k in KEY},
                         label=int(row["label"]), probability=float(prob), reference=float(row["reference"]),
                         outcome=float(row["outcome"]), animals=int(row["animals"]),
                         reference_dates=int(row["reference_dates"]), shift_low=lo, shift_high=hi, significant=sig))
    return rows


def summarize(frame):
    rows = []
    for significant in [False, True]:
        selected = frame[frame["significant"].eq(True)] if significant else frame
        for fold, group in selected.groupby("fold"):
            rows.append(dict(significant=significant, fold=fold, groups=len(group),
                             dates=int(group["kill date"].nunique()), **metrics(group, group["probability"])))
    result = {}
    for significant, suffix in [(False, "all"), (True, "significant")]:
        chosen = [row for row in rows if row["significant"] == significant]
        contributing = [row for row in chosen if row["pair_weight"] > 0]
        pair = sum(row["pair_weight"] for row in contributing)
        result.update({f"auc_{suffix}": sum(row["auc"]*row["pair_weight"] for row in contributing)/pair if pair else None,
                       f"evaluated_groups_{suffix}": sum(row["groups"] for row in chosen),
                       f"evaluated_dates_{suffix}": sum(row["dates"] for row in chosen),
                       f"contributing_folds_{suffix}": len(contributing),
                       f"contributing_groups_{suffix}": sum(row["groups"] for row in contributing),
                       f"contributing_dates_{suffix}": sum(row["dates"] for row in contributing)})
        for metric in ["log_loss", "brier_score"]:
            result[f"{metric}_{suffix}"] = (float(np.average([row[metric] for row in chosen],
                                                          weights=[row["dates"] for row in chosen])) if chosen else None)
    return result, rows


def validate_prediction_sets(predictions):
    if not predictions:
        return
    frame = pd.DataFrame(predictions)
    if frame.duplicated(FIT_KEY + KEY).any():
        raise ValueError("Duplicate saved predictions.")
    group_key = ["animal_type", "target", "fold"] + KEY
    for col in ["label", "reference", "outcome", "significant", "shift_low", "shift_high"]:
        if frame.groupby(group_key)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent held-out {col} across jobs.")
    if frame.groupby(["animal_type", "target", "kill date"])["fold"].nunique().gt(1).any():
        raise ValueError("Date appears in multiple outer test folds.")
    for _, group in frame.groupby(["animal_type", "target", "fold"]):
        sets = [set(map(tuple, part[KEY].to_numpy())) for _, part in group.groupby(["model", "setting", "horizon"])]
        if any(value != sets[0] for value in sets):
            raise ValueError("Jobs have different held-out groups.")


def verify_significance(plans):
    # Independently reconstruct the reporting filter from the frozen outcomes.
    for animal, target, fold, train, _, held in plans:
        intervals = significance(train, held, np.random.default_rng(SEED + fold))
        expected = records_for_fit({}, held, np.zeros(len(held)), intervals)
        for name, setting, horizon in jobs():
            info = dict(zip(FIT_KEY, [animal, target, name, setting, horizon, fold]))
            path = fit_path(info)
            if not path.exists():
                continue
            fit = json.loads(path.read_text())
            if "failure" in fit:
                continue
            actual = fit["predictions"]
            if len(actual) != len(expected):
                raise ValueError(f"Significance row count differs: {path}")
            for row, reference in zip(actual, expected):
                for key in KEY + ["animals", "reference_dates", "significant"]:
                    if row[key] != reference[key]:
                        raise ValueError(f"Saved {key} differs from the significance audit: {path}")
                for key in ["shift_low", "shift_high"]:
                    if row[key] is None or reference[key] is None:
                        equal = row[key] is reference[key]
                    else:
                        equal = np.isclose(row[key], reference[key], rtol=1e-12, atol=1e-12)
                    if not equal:
                        raise ValueError(f"Saved {key} differs from the bootstrap audit: {path}")


def export_results(plans):
    predictions, scores, candidates, features, failures, status = [], [], [], [], [], []
    complete = {}
    expected_paths = {fit_path(dict(zip(FIT_KEY, [animal, target, name, setting, horizon, fold])))
                      for animal, target, fold, *_ in plans for name, setting, horizon in jobs()}
    if set((OUT / "fits").glob("*.json")) - expected_paths:
        raise ValueError("Unexpected fit checkpoints in output directory; do not mix runs.")
    for animal, target, fold, train, a, b in plans:
        for name, setting, horizon in jobs():
            info = dict(zip(FIT_KEY, [animal, target, name, setting, horizon, fold]))
            path = fit_path(info)
            if not path.exists():
                status.append(dict(**info, status="pending"))
                continue
            fit = json.loads(path.read_text())
            if any(fit[key] != info[key] for key in FIT_KEY):
                raise ValueError(f"Checkpoint key mismatch: {path}")
            candidates.extend({**row, **info, "params": json.dumps(row["params"]),
                               "selected_weather": json.dumps(row.get("selected_weather", []))}
                              for row in fit.get("candidates", []))
            if "failure" in fit:
                failures.append(dict(**info, stage=fit["stage"], error=fit["failure"]))
                status.append(dict(**info, status="failed"))
                continue
            frame = pd.DataFrame(fit["predictions"])
            expected = b.copy()
            expected["kill date"] = expected["kill date"].map(str)
            if len(frame) != len(b) or set(map(tuple, frame[KEY].to_numpy())) != set(map(tuple, expected[KEY].to_numpy())):
                raise ValueError(f"Saved group membership changed: {path}")
            check = frame.merge(expected[KEY + ["label", "reference", "outcome"]], on=KEY, suffixes=("", "_expected"), validate="one_to_one")
            for col in ["label", "reference", "outcome"]:
                if not np.allclose(check[col], check[col + "_expected"], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Saved {col} differs from recomputed labels: {path}")
            prob = frame["probability"]
            if not np.isfinite(prob).all() or not prob.between(0, 1).all():
                raise ValueError(f"Invalid saved probabilities: {path}")
            predictions.extend(fit["predictions"])
            score, _ = summarize(frame)
            scores.append(dict(**info, **score, best_params=json.dumps(fit["best_params"]),
                               n_features=len(fit["feature_names"]), n_weather=len(fit["selected_weather"]),
                               nonzero_coefficients=fit["nonzero_coefficients"],
                               probability_min=float(prob.min()), probability_max=float(prob.max())))
            features.extend(dict(**info, feature=feature) for feature in fit["selected_weather"])
            status.append(dict(**info, status="complete"))
            complete.setdefault(tuple(info[key] for key in CELL_KEY), []).extend(fit["predictions"])
    validate_prediction_sets(predictions)
    status_frame = pd.DataFrame(status)
    table, fold_metrics = [], []
    for key, group in status_frame.groupby(CELL_KEY, sort=False):
        info = dict(zip(CELL_KEY, key))
        done = int(group["status"].eq("complete").sum())
        state = "complete" if done == N_OUTER else "incomplete"
        # Never present a partial-fold aggregate as a paper result.
        entry = dict(**info, slice="overall", status=state, completed_folds=done, expected_folds=N_OUTER)
        if state != "complete":
            table.append(entry)
            continue
        frame = pd.DataFrame(complete[key])
        slices = [("overall", frame)] + [(f"teeth={int(teeth)}", part) for teeth, part in frame.groupby("Teeth")]
        for label, part in slices:
            summary, rows = summarize(part)
            table.append({**entry, "slice": label, **summary})
            fold_metrics.extend(dict(**info, slice=label, **row) for row in rows)
    save_csv(predictions, "predictions.csv")
    save_csv(scores, "fold_scores.csv")
    save_csv(pd.DataFrame(failures, columns=FIT_KEY + ["stage", "error"]), "failed_fits.csv")
    save_csv(status, "fit_status.csv")
    save_csv(candidates, "candidate_diagnostics.csv")
    save_csv(features, "selected_weather.csv")
    save_csv(fold_metrics, "fold_metrics.csv")
    save_csv(table, "result_table.csv")
    result = dict(expected=len(status), complete=len(scores), failed=len(failures),
                  pending=len(status)-len(scores)-len(failures), prediction_checks_passed=True)
    save_json(OUT / "run_status.json", result)
    return result


def run_experiments(data, weather, plans):
    (OUT / "fits").mkdir(exist_ok=True)
    state = export_results(plans)
    verify_significance(plans)
    tqdm.write(f"Resuming: {state['complete']}/{state['expected']} outer fits already complete.")
    try:
        with tqdm(total=state["expected"], desc="Outer fits") as progress:
            for animal, target, fold, train, a, b in plans:
                intervals = None
                for name, setting, horizon in jobs():
                    info = dict(zip(FIT_KEY, [animal, target, name, setting, horizon, fold]))
                    path = fit_path(info)
                    if path.exists() and "failure" not in json.loads(path.read_text()):
                        progress.update()
                        continue
                    progress.set_postfix_str(f"{animal}, {target}, {name}, {horizon}d, fold {fold}, {setting}")
                    result = dict(**info, candidates=[])
                    stage = "tuning"
                    try:
                        params, diagnostics = tune(train, weather, name, setting, horizon)
                        result["candidates"] = diagnostics
                        if params is None:
                            raise ValueError("No candidate succeeded in every inner fit with evaluable AUC.")
                        stage = "refit"
                        probability, details = fit_predict(a, b, weather, name, setting, horizon, params)
                        stage = "evaluation"
                        if intervals is None:
                            intervals = significance(train, b, np.random.default_rng(SEED + fold))
                        result.update(best_params=params, **details,
                                      predictions=records_for_fit(info, b, probability, intervals))
                    except Exception as error:
                        result.update(failure=repr(error), stage=stage)
                        tqdm.write(f"Failed: {info}: {error}")
                    save_json(path, result)
                    export_results(plans)
                    progress.update()
    except KeyboardInterrupt:
        export_results(plans)
        tqdm.write("Interrupted. Completed fits are saved; rerun the same command to resume.")
        return
    state = export_results(plans)
    verify_significance(plans)
    print(f"Finished: {state['complete']}/{state['expected']} fits complete; {state['failed']} failed.")
    if LOCKED_EXPORTS:
        print("Some CSVs were locked. Close them and rerun to refresh exports from checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true", help="Verify and refresh results without fitting; require a complete run.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data, weather = prepare_data()
        plans = describe_folds(data, weather)
        if args.prepare_only or args.verify_only:
            state = export_results(plans)
            if args.verify_only:
                verify_significance(plans)
            print(f"{state['complete']}/{state['expected']} complete; {state['failed']} failed; {state['pending']} pending. Output: {OUT}")
            if args.verify_only and state["complete"] != state["expected"]:
                raise SystemExit(1)
        else:
            run_experiments(data, weather, plans)



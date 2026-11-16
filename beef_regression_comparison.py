"""Fixed-horizon regression comparison. Run with --prepare-only to check inputs."""
import argparse
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

# Single-worker fits do not need joblib's Windows physical-core probe.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/beef_regression_comparison_10fold_animal'
N_OUTER = 10
TARGETS = ['live weight', 'carcass weight', 'price']
ANIMALS = ['heifer', 'steer']
HORIZONS = [30, 60, 120, 180]
SEED = 2210
KEY = ['animal_type', 'target', 'model', 'setting', 'horizon', 'pruning', 'fold']


def specs():
    # Previous fits favoured smaller trees and slower boosting. Horizons are not tuned.
    trees = dict(n_estimators=300, max_features=0.5, n_jobs=1, random_state=SEED)
    return {
        'ElasticNet': (ElasticNet(max_iter=50000), {'alpha': [0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5]}),
        'RandomForest': (RandomForestRegressor(**trees), {'min_samples_leaf': [5, 15], 'max_depth': [6, None]}),
        'ExtraTrees': (ExtraTreesRegressor(**trees), {'min_samples_leaf': [5, 15], 'max_depth': [6, None]}),
        'LightGBM': (LGBMRegressor(n_estimators=200, learning_rate=0.03, n_jobs=1,
                         verbosity=-1, deterministic=True, force_col_wise=True, random_state=SEED),
                     {'num_leaves': [7, 15], 'min_child_samples': [15, 30]}),
        'HistGradientBoosting': (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03,
                                   l2_regularization=1, early_stopping=False, random_state=SEED),
                                {'max_leaf_nodes': [7, 15], 'min_samples_leaf': [15, 30]})}


def jobs():
    yield 'Dummy', 'Dummy', 0, 'none'
    for model in specs():
        yield model, 'Teeth only', 0, 'none'
        yield model, 'Teeth + NZDI', 0, 'none'
        for horizon in HORIZONS:
            for pruning in ['all', 'spearman']:
                yield model, 'Teeth + weather', horizon, pruning


def weights(data):
    # Equal-date weights for the secondary date-MAE summaries.
    w = 1 / data.groupby('kill date')['kill date'].transform('size').to_numpy()
    return w / w.mean()


def splits(data, n):
    dates = np.sort(data['kill date'].unique())
    for a, b in KFold(n, shuffle=True, random_state=SEED).split(dates):
        yield data[data['kill date'].isin(dates[a])], data[data['kill date'].isin(dates[b])]


def selected_weather(train, columns, horizon, pruning):
    columns = [c for c in columns if int(c.split('__win')[1].split('_')[1]) <= horizon]
    if pruning == 'all':
        return columns
    # One observation per training date. Exceedance counts are treated like all other features.
    dates = train.groupby('kill date')[columns].mean()
    usable = [c for c in columns if dates[c].nunique() > 1]
    corr = dates[usable].corr(method='spearman').abs()
    kept = []
    for c in sorted(usable):
        if not kept or not corr.loc[c, kept].ge(0.95).any():
            kept.append(c)
    return kept


def predict(train, test, target, model, setting, horizon, pruning, params, weather):
    w = np.ones(len(train))  # Equal animal weights for fitting, scaling and the dummy.
    if model == 'Dummy':
        return np.full(len(test), np.average(train[target], weights=w)), []
    numeric = [] if setting == 'Teeth only' else ['NZDI']
    if setting == 'Teeth + weather':
        numeric += selected_weather(train, weather, horizon, pruning)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    a = encoder.fit_transform(train[['Teeth category']])
    b = encoder.transform(test[['Teeth category']])
    if numeric:
        scaler = StandardScaler().fit(train[numeric], sample_weight=w)
        a = np.column_stack([a, scaler.transform(train[numeric])])
        b = np.column_stack([b, scaler.transform(test[numeric])])
    estimator = clone(specs()[model][0]).set_params(**params)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(a, train[target], sample_weight=w)
        prediction = estimator.predict(b)
    return prediction, numeric


def tune(train, target, model, setting, horizon, pruning, weather):
    if model == 'Dummy':
        return {}, []
    diagnostics, choices = [], []
    for candidate, params in enumerate(ParameterGrid(specs()[model][1])):
        losses, ok = [], True
        for inner, (a, b) in enumerate(splits(train, 3), 1):
            row = dict(candidate=candidate, params=params, inner=inner)
            try:
                pred, columns = predict(a, b, target, model, setting, horizon, pruning, params, weather)
                errors = np.abs(b[target].to_numpy() - pred)
                losses.extend(errors.tolist())
                row.update(mae=float(errors.mean()), animals=len(b), dates=b['kill date'].nunique(), selected_numeric=columns)
            except Exception as error:
                row['error'] = repr(error)
                ok = False
            diagnostics.append(row)
        if ok:
            choices.append((float(np.mean(losses)), candidate, params))
    if not choices:
        return None, diagnostics
    return min(choices, key=lambda x: x[:2])[2], diagnostics


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT / 'data/beef_preprocessed.csv', ROOT / 'data/beef_weather_windows.csv']
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest['versions'] = dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__)
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Code or inputs changed; use a new output directory.')
    write_json(path, manifest)
    beef = pd.read_csv(paths[1], parse_dates=['kill date'])
    windows = pd.read_csv(paths[2])
    data = beef.join(windows)
    data['animal_index'] = data.index
    data = data.loc[data['Animal Type'].isin(ANIMALS)].copy()
    weather = list(windows)
    if data[weather + ['NZDI', 'Teeth category']].isna().any().any():
        raise ValueError('Missing predictors: inspect inputs before adding a training-fold imputer.')
    plans, counts = [], []
    for animal in ANIMALS:
        for target in TARGETS:
            sub = data.loc[data['Animal Type'].eq(animal)].dropna(subset=[target])
            for fold, (train, test) in enumerate(splits(sub, N_OUTER), 1):
                plans.append((animal, target, fold, train, test))
                for role, part in [('train', train), ('test', test)]:
                    counts.append(dict(animal_type=animal, target=target, fold=fold, role=role,
                                       animals=len(part), dates=part['kill date'].nunique()))
    pd.DataFrame(counts).to_csv(OUT / 'fold_counts.csv', index=False)
    membership = [dict(animal_type=a, target=t, fold=f, animal_index=int(row.animal_index),
                       kill_date=str(row['kill date'])) for a,t,f,_,test in plans for _,row in test.iterrows()]
    pd.DataFrame(membership).to_csv(OUT / 'fold_mapping.csv', index=False)
    return weather, plans


def oracle(plans):
    # Hindsight reference only: held-out date/teeth outcomes define their own median.
    rows = []
    for animal in ANIMALS:
        for target in TARGETS:
            parts = [test for a,t,_,_,test in plans if a == animal and t == target]
            if not parts:
                continue
            data = pd.concat(parts)
            medians = data.groupby(['kill date', 'Teeth category'])[target].transform('median')
            errors = (data[target] - medians).abs()
            rows.append(dict(animal_type=animal, target=target, model='Date-teeth oracle',
                             evaluation='Oracle (uses observed outcomes)',
                             date_mae=float(np.average(errors, weights=weights(data))),
                             animal_mae=float(errors.mean())))
    pd.DataFrame(rows).to_csv(OUT / 'oracle_scores.csv', index=False)


def export():
    fits = [json.loads(p.read_text()) for p in sorted((OUT / 'fits').glob('*.json'))]
    if not fits:
        return
    predictions = pd.DataFrame([r for f in fits for r in f['predictions']])
    predictions['error'] = (predictions.observed - predictions.predicted).abs()
    rows = []
    for key, group in predictions.groupby(KEY[:-1]):
        if group.fold.nunique() == N_OUTER:
            rows.append({**dict(zip(KEY[:-1], key)), 'date_mae': group.groupby('kill_date').error.mean().mean(),
                         'animal_mae': group.error.mean(), 'animals': len(group), 'dates': group.kill_date.nunique()})
    # JSON fit checkpoints remain authoritative if a CSV reader locks an export.
    outputs = {'predictions.csv': predictions, 'scores.csv': pd.DataFrame(rows),
               'fold_scores.csv': pd.DataFrame([f['score'] for f in fits]),
               'candidate_diagnostics.csv': pd.DataFrame([r for f in fits for r in f['diagnostics']])}
    for name, frame in outputs.items():
        temporary = OUT / (name + '.tmp')
        frame.to_csv(temporary, index=False)
        try:
            temporary.replace(OUT / name)
        except PermissionError:
            tqdm.write(f'{name} locked; checkpoints retained. Rerun to refresh exports.')


def run(weather, plans):
    folder = OUT / 'fits'
    folder.mkdir(exist_ok=True)
    failures = OUT / 'failures'
    failures.mkdir(exist_ok=True)
    with tqdm(total=len(plans)*len(list(jobs())), desc='Outer fits') as bar:
        for animal, target, fold, train, test in plans:
            for model, setting, horizon, pruning in jobs():
                info = dict(zip(KEY, [animal, target, model, setting, horizon, pruning, fold]))
                name = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest() + '.json'
                path = folder / name
                if not path.exists():
                    bar.set_postfix_str(f'{animal}, {target}, {model}, {horizon}d, {pruning}, fold {fold}')
                    diagnostics = []
                    try:
                        params, diagnostics = tune(train, target, model, setting, horizon, pruning, weather)
                        if params is None:
                            raise ValueError('No candidate succeeded in all inner folds.')
                        pred, columns = predict(train, test, target, model, setting, horizon, pruning, params, weather)
                        error = np.abs(test[target].to_numpy()-pred)
                        rows = [{**info, 'animal_index': int(idx), 'kill_date': str(date),
                                 'observed': float(y), 'predicted': float(p)} for idx,date,y,p in
                                zip(test.animal_index, test['kill date'], test[target], pred)]
                        score = {**info, 'date_mae': float(np.average(error, weights=weights(test))),
                                 'animal_mae': float(error.mean()), 'params': params, 'selected_numeric': columns}
                        write_json(path, dict(score=score, predictions=rows,
                                              diagnostics=[{**info, **d} for d in diagnostics]))
                        (failures / name).unlink(missing_ok=True)
                    except Exception as error:
                        write_json(failures / name, dict(**info, error=repr(error), diagnostics=diagnostics))
                bar.update()
    export()
    print(f'Completed {len(list(folder.glob("*.json")))} / {len(plans)*len(list(jobs()))}; '
          f'{len(list(failures.glob("*.json")))} failures pending.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    weather, plans = prepare()
    oracle(plans)
    if args.prepare_only:
        print(f'Prepared {len(plans)} outer splits and {len(plans)*len(list(jobs()))} fits; no missing predictors.')
    else:
        run(weather, plans)

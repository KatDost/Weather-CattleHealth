# Weather-Impacts-on-Cattle-Growth-Health-Productivity

Installing the required packages:
```
python -m pip install pip-tools
python -m piptools compile requirements.in
python -m pip install -r requirements.txt
```

## Beef experiments

Run `python beef_experiments.py` from the project environment. Use
`python beef_experiments.py --prepare-only` to prepare the fold mapping and check
all classification splits without fitting models.

The runner excludes cows and animals with estimated age below 120 days (four
30-day months). It retains the existing weather windows through 60 days and
adds 61–90 and 91–120 days using the same summaries and whole-series thresholds.
There is no weather-coverage exclusion. Missing predictors are imputed within CV.

ElasticNet regression, elastic-net logistic regression, SVR/SVC, random forests,
LightGBM and histogram gradient boosting use ten fixed outer
folds and three inner folds, both keeping slaughter dates together. The saved
mapping balances animal counts separately by animal type and is reused on later
runs. Each target excludes missing outcomes. Non-weather-only and dummy baselines
run once per target; weather-only and combined features run at all three horizons.

Outputs are in `results/beef_experiments/`: the fold mapping, appendix fold counts,
fold scores and parameters, held-out predictions, and pooled performance scores.
Scores and predictions are checkpointed after each fit. Rerunning skips completed
animal/target/model/feature-setting/fold combinations and preserves their results.
Progress counts outer fits, including their inner tuning and already completed
fits. Settings and parallel worker count are near the top of the script.

To add models, add named estimator/grid pairs to `model_specs()` and rerun. All
models use the same imputation/scaling pipeline; grid keys start with `model__`.
The dummy baseline is shared and completed ElasticNet fits are not repeated.
Saved model names identify experiments: use a new name if changing an estimator
or its tuning grid. If changing the data, features, folds or preprocessing, use
a fresh output directory instead of mixing results from different designs.
Run one instance at a time against a given output directory. Pooled scores are
reported only for combinations with all ten outer folds completed.

If Windows locks an output CSV, saving retries briefly, then writes a uniquely
named fallback in `checkpoints/` and continues. Resume automatically loads the
newest snapshot for each output. Close applications holding the CSVs and rerun to
refresh the usual filenames without repeating completed fits. Disk-write failures
(for example a full drive) are still reported rather than silently losing results.

Non-convergence invalidates a tuning candidate if it occurs in any inner fold.
Failed searches or final refits are recorded in `failed_fits.csv`, and remaining
jobs continue. Rerunning retries pending jobs while skipping successful ones;
unchanged solver settings may produce the same failures, so review that file
before retrying. Successful scores include the number of rejected candidates.
Earlier results that allowed unconverged candidates are backed up with the suffix
`_before_convergence_rejection.csv` and recomputed under the new policy.

The additional models use fixed compact grids: 24 SVR / 12 SVC candidates,
12 random-forest candidates, and 16 candidates each for LightGBM and histogram
gradient boosting. Classification models use balanced class weights. Automatic
early stopping is disabled for histogram boosting to avoid random internal
validation splits that ignore slaughter dates. Candidate searches use `N_JOBS`
workers with one native thread per worker; final refits also use one thread.
The full design adds 2,240 outer fits to the 640 initial ElasticNet/baseline fits.

The revised non-weather comparator contains age, transport distance and lairage
time, excluding NZDI. Combined predictors still include NZDI. Weather-only results
retain their original definition (window summaries without NZDI) and were not
rerun in this revision. Thus combined versus non-weather comparisons measure the
added value of weather windows together with NZDI.

Previous beef results and figures are archived in `results/old/` and `figs/old/`.
Existing analysis notebooks still reference the previous output paths and need
updating before they can plot the new experiment results.

"""Final regression checks for the paper comparison runner.

Run: python -m unittest discover -s tests -p test_beef_direction_experiments.py
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import beef_direction_experiments_10fold as experiment


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.groups = pd.DataFrame({
            "kill date": pd.date_range("2020-01-01", periods=12),
            "Teeth": [0] * 12, "Animal Type": ["steer"] * 12,
            "category": ["0/steer"] * 12, "animals": [2] * 12,
            "outcome": [10., 20., 60.] * 4,
            "values": [[10., 10.], [20., 20.], [60., 60.]] * 4,
            "NZDI": np.linspace(0, 1, 12),
            "wind__win0_4_mean": np.arange(12, dtype=float),
            "temperature__win31_60_mean": np.arange(12, dtype=float)[::-1],
            "rain__win121_180_mean": np.arange(12, dtype=float) ** 2})
        self.weather = [col for col in self.groups if "__win" in col]

    def test_reference_labels_ties_and_unseen_categories(self):
        train = self.groups.iloc[:2]
        np.testing.assert_array_equal(experiment.labels(train)["reference"], [20., 10.])
        held = self.groups.iloc[2:3].copy()
        self.assertEqual(experiment.labels(train, held)["reference"].iloc[0], 15.)
        held["outcome"] = 15.
        self.assertEqual(experiment.labels(train, held)["exclusion"].iloc[0], "tie")
        held["Teeth"] = 8
        self.assertEqual(experiment.labels(train, held)["exclusion"].iloc[0], "no_reference")
        self.assertTrue(experiment.eligible(experiment.labels(train.iloc[:1])).empty)

    def test_date_separation_and_weights(self):
        frame = pd.concat([self.groups, self.groups.iloc[:1]], ignore_index=True)
        w = pd.Series(experiment.weights(frame)).groupby(frame["kill date"]).sum()
        np.testing.assert_array_equal(w, np.ones(12))
        for train, test in experiment.splits(frame, 5):
            self.assertFalse(set(train["kill date"]) & set(test["kill date"]))

    def test_full_model_horizon_grid(self):
        jobs = list(experiment.jobs())
        self.assertEqual(len(jobs), 31)
        self.assertEqual(len(set(jobs)), 31)
        for name in experiment.model_specs():
            self.assertEqual([h for n, s, h in jobs if n == name and s == "weather_pruned"], [30, 60, 120, 180])

    def test_loading_includes_full_period_exceedance_counts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "ROOT", Path(directory)):
            source = Path(directory) / "data"
            source.mkdir(parents=True)
            self.groups[["kill date", "Animal Type", "Teeth"]].to_csv(source / "beef_preprocessed.csv", index=False)
            columns = self.groups[self.weather].copy()
            count = "MnTemp(C)__win0_4_above25.0_count"
            columns[count] = np.arange(len(columns))
            columns.to_csv(source / "beef_weather_windows.csv", index=False)
            data, weather = experiment.load_data(Path(directory) / "output")
            self.assertIn(count, weather)
            np.testing.assert_array_equal(data[count], columns[count])

    def test_horizon_precedes_pruning_and_pruning_ignores_labels(self):
        frame = self.groups.assign(label=[0, 1] * 6)
        self.assertEqual(experiment.select_weather(frame, self.weather, 30), [self.weather[0]])
        kept = experiment.select_weather(frame, self.weather, 60)
        self.assertEqual(len(kept), 1)  # perfectly correlated short/medium summaries
        self.assertNotIn(self.weather[2], kept)
        self.assertEqual(kept, experiment.select_weather(frame.assign(label=1-frame.label), self.weather, 60))
        missing = frame.copy()
        missing[self.weather] = np.nan
        self.assertEqual(experiment.select_weather(missing, self.weather, 180), [])

    def test_all_models_probability_api_and_training_only_features(self):
        train = experiment.eligible(experiment.labels(self.groups))
        held = train.iloc[:3].copy()
        with threadpool_limits(limits=1):
            for name, (_, grid) in experiment.model_specs().items():
                probability, details = experiment.fit_predict(train, held, self.weather, name, "weather_pruned", 30, grid[0])
                changed = held.assign(label=1-held.label, significant=True)
                other, other_details = experiment.fit_predict(train, changed, self.weather, name, "weather_pruned", 30, grid[0])
                np.testing.assert_array_equal(probability, other)
                self.assertEqual(details["selected_weather"], [self.weather[0]])
                self.assertEqual(details, other_details)
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_animal_types_cannot_be_pooled(self):
        frame = experiment.eligible(experiment.labels(self.groups))
        frame.loc[0, "Animal Type"] = "heifer"
        with self.assertRaisesRegex(ValueError, "animal type"):
            experiment.fit_predict(frame, frame, self.weather, "ElasticNet", "controls", 0, {"C": .1})

    def test_auc_stays_within_fold_and_significance_only_filters_reporting(self):
        rows = []
        for fold, labels in [(1, [0, 0, 1]), (2, [0, 1, 1])]:
            for i, label in enumerate(labels):
                rows.append(dict(fold=fold, label=label, probability=.2 if fold == 1 else .8,
                                 significant=True, **{"kill date": f"2020-0{fold}-0{i+1}"}))
        frame = pd.DataFrame(rows)
        summary, _ = experiment.summarize(frame)
        self.assertEqual(summary["auc_all"], .5)
        self.assertEqual(summary["auc_significant"], .5)
        other, _ = experiment.summarize(frame.assign(significant=False))
        self.assertEqual(other["auc_all"], .5)
        self.assertIsNone(other["auc_significant"])
        self.assertEqual(other["evaluated_groups_significant"], 0)

    def test_bootstrap_minimum_support(self):
        result = experiment.significance(self.groups.iloc[:2], self.groups.iloc[2:3], np.random.default_rng(1))
        self.assertTrue(result[0][2])
        result = experiment.significance(self.groups.iloc[:1], self.groups.iloc[2:3], np.random.default_rng(1))
        self.assertEqual(result[0], (None, None, None))

    def test_tuning_keeps_model_and_horizon_fixed(self):
        actual = experiment.fit_predict
        calls = []
        def capture(train, test, weather, name, setting, horizon, params):
            calls.append((name, horizon))
            return actual(train, test, weather, name, setting, horizon, params)
        with patch.object(experiment, "fit_predict", side_effect=capture), threadpool_limits(limits=1):
            params, diagnostics = experiment.tune(self.groups, self.weather, "ElasticNet", "weather_pruned", 30)
        self.assertIsNotNone(params)
        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(set(calls), {("ElasticNet", 30)})
        self.assertTrue(all("failure" not in row for row in diagnostics))

    def test_interruption_resume_complete_tables_and_corruption_detection(self):
        train = self.groups.iloc[:9]
        a = experiment.eligible(experiment.labels(train))
        b = experiment.eligible(experiment.labels(train, self.groups.iloc[9:]))
        plans = [("steer", "weight", 1, train, a, b)]
        selected_jobs = [("Dummy", "dummy", 0), ("ElasticNet", "weather_pruned", 30)]
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "OUT", Path(directory)), \
             patch.object(experiment, "N_OUTER", 1), patch.object(experiment, "jobs", return_value=selected_jobs):
            with patch.object(experiment, "tune", side_effect=[({}, []), KeyboardInterrupt]):
                experiment.run_experiments(None, self.weather, plans)
            table = pd.read_csv(Path(directory) / "result_table.csv")
            overall = table[table["slice"].eq("overall")]
            self.assertEqual(set(overall.status), {"complete", "incomplete"})
            saved = next((Path(directory) / "fits").glob("*Dummy*.json"))
            original = saved.read_bytes()
            with patch.object(experiment, "tune", return_value=({"C": .1}, [])) as tuning:
                experiment.run_experiments(None, self.weather, plans)
            self.assertEqual(tuning.call_count, 1)
            self.assertEqual(saved.read_bytes(), original)
            status = experiment.export_results(plans)
            self.assertEqual(status["complete"], 2)
            experiment.verify_significance(plans)
            fit = json.loads(saved.read_text())
            fit["predictions"][0]["label"] = 1-fit["predictions"][0]["label"]
            saved.write_text(json.dumps(fit))
            with self.assertRaisesRegex(ValueError, "label"):
                experiment.export_results(plans)

    def test_manifest_rejects_changed_design(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "OUT", Path(directory)), \
             patch.object(experiment, "load_data", return_value=(self.groups, self.weather)):
            (Path(directory) / "inputs").mkdir()
            (Path(directory) / "inputs/manifest.json").write_text("{}")
            experiment.prepare_data()
            with patch.object(experiment, "CORRELATION_LIMIT", .9):
                with self.assertRaisesRegex(ValueError, "design changed"):
                    experiment.prepare_data()


if __name__ == "__main__":
    unittest.main()

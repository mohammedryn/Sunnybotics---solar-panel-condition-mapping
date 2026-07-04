"""
Tests for the supervised external clean/damaged benchmark
(src/external_classifier.py): held-out cross-validation correctness,
summary/importance/confidence-strata output shape, and the structural
guarantee that folder labels never leak into the normal, label-free
inference pipeline.

Run with: python3 -m unittest tests.test_external_classifier -v
"""
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import condition_analysis, external_classifier as ec


def _features(dirt=0.0, shadow=0.0, glare=0.0, damage_density=0.0, damage_irregular=0.0,
              brightness=0.5, hue_shift=0.0, clipped=0.0, directionality=0.0, laplacian=500.0):
    return {
        "brightness_median": brightness,
        "shadow_area_ratio": shadow,
        "shadow_directionality_score": directionality,
        "dirt_area_ratio": dirt,
        "dirt_hue_shift_score": hue_shift,
        "glare_area_ratio": glare,
        "glare_clipped_pixel_ratio": clipped,
        "damage_line_density": damage_density,
        "damage_irregular_line_score": damage_irregular,
        "laplacian_variance": laplacian,
    }


class TestStratifiedKFold(unittest.TestCase):
    def test_folds_are_disjoint_and_cover_every_index_exactly_once(self):
        y = np.array(["clean"] * 10 + ["damaged"] * 10)
        folds = ec._stratified_kfold_indices(y, n_folds=5, seed=42)
        seen = []
        for train_idx, test_idx in folds:
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
            seen.extend(test_idx.tolist())
        self.assertEqual(sorted(seen), list(range(len(y))))

    def test_each_fold_test_set_is_stratified_across_classes(self):
        y = np.array(["clean"] * 10 + ["damaged"] * 10)
        folds = ec._stratified_kfold_indices(y, n_folds=5, seed=42)
        for _, test_idx in folds:
            test_labels = y[test_idx]
            self.assertIn("clean", test_labels)
            self.assertIn("damaged", test_labels)


class TestCrossValidateHeldOut(unittest.TestCase):
    def test_every_prediction_is_held_out_and_separable_data_scores_well(self):
        """damage_line_density alone perfectly separates the two classes
        here - if cross_validate were leaking test rows into their own
        training fold, this would trivially score 100%; the real
        assertion that matters is TestStratifiedKFold's disjointness
        check above. This test just confirms the wiring produces a
        sensible, non-degenerate result on an easy, separable case."""
        rows = []
        for i in range(15):
            rows.append({"image_id": f"clean{i}", "true_label": "clean", **_features(damage_density=0.001)})
        for i in range(15):
            rows.append({"image_id": f"dmg{i}", "true_label": "damaged", **_features(damage_density=0.5)})
        feature_table = pd.DataFrame(rows)

        cv_eval_df = ec.cross_validate(feature_table)
        self.assertEqual(len(cv_eval_df), 30)
        self.assertFalse(cv_eval_df["cv_predicted_label"].isna().any())
        self.assertFalse(cv_eval_df["cv_confidence"].isna().any())
        self.assertGreater(cv_eval_df["cv_correct"].mean(), 0.8)

    def test_predict_step_has_no_parameter_slot_for_a_label_at_all(self):
        """Stronger than a source-text scan: the function that produces
        predictions during cross-validation has a fixed 2-parameter
        signature (model, X) - there is no argument position a label
        could occupy even if some future edit tried to pass one in."""
        sig = inspect.signature(ec._predict_with_confidence)
        self.assertEqual(list(sig.parameters.keys()), ["model", "X"])

    def test_fit_only_ever_sees_its_own_folds_train_rows_and_predict_gets_no_labels(self):
        """Behavioral proof, not just a signature/text check: spies on
        the real _fit_backend/_predict_with_confidence calls made inside
        cross_validate (via unittest.mock.patch.object with wraps=, so
        the real computation still runs) and asserts, per fold: (1) the
        rows handed to fit and the rows handed to predict are completely
        disjoint (no row is trained AND predicted on in the same fold),
        (2) fit's train set size plus predict's test set size equals the
        whole dataset (a true partition, not a subset missing some rows),
        and (3) the predict call never carries a labels argument - it's
        always exactly (model, X_test), nothing more. Each row's feature
        vector is made unique (via a per-row brightness nudge) so
        comparing raw row tuples is a valid way to prove index-level
        disjointness, not just "these two DataFrames happen to look
        different"."""
        rows = []
        for i in range(30):
            label = "clean" if i < 15 else "damaged"
            rows.append({
                "image_id": f"img{i}", "true_label": label,
                **_features(damage_density=0.5 if label == "damaged" else 0.001, brightness=0.5 + i * 0.001),
            })
        feature_table = pd.DataFrame(rows)

        with patch.object(ec, "_fit_backend", wraps=ec._fit_backend) as fit_spy, \
             patch.object(ec, "_predict_with_confidence", wraps=ec._predict_with_confidence) as predict_spy:
            ec.cross_validate(feature_table)

        self.assertGreater(fit_spy.call_count, 0)
        self.assertEqual(fit_spy.call_count, predict_spy.call_count)

        for fit_call, predict_call in zip(fit_spy.call_args_list, predict_spy.call_args_list):
            self.assertEqual(len(fit_call.args), 2)
            X_train_seen, y_train_seen = fit_call.args

            self.assertEqual(len(predict_call.args), 2)  # exactly (model, X) - never a 3rd/label arg
            self.assertEqual(predict_call.kwargs, {})
            _, X_test_seen = predict_call.args

            self.assertEqual(len(y_train_seen), len(X_train_seen))  # fit's labels match its own rows, no more
            self.assertEqual(len(X_train_seen) + len(X_test_seen), len(feature_table))  # true partition

            train_rows = {tuple(row) for row in X_train_seen}
            test_rows = {tuple(row) for row in X_test_seen}
            self.assertEqual(train_rows & test_rows, set())  # no row is both trained-on and predicted-on


class TestFinalModelAndImportance(unittest.TestCase):
    def _table(self):
        rows = []
        for i in range(15):
            rows.append({"image_id": f"clean{i}", "true_label": "clean", **_features(damage_density=0.001, dirt=0.01)})
        for i in range(15):
            rows.append({"image_id": f"dmg{i}", "true_label": "damaged", **_features(damage_density=0.5, dirt=0.01)})
        return pd.DataFrame(rows)

    def test_final_model_predicts_on_all_rows(self):
        table = self._table()
        model_info = ec.fit_final_model(table)
        preds = ec.predict_with_final_model(model_info, table)
        self.assertEqual(len(preds), len(table))
        self.assertIn("external_predicted_label", preds.columns)
        self.assertIn("external_confidence", preds.columns)

    def test_feature_importance_sorted_descending_by_abs_coefficient(self):
        table = self._table()
        model_info = ec.fit_final_model(table)
        importance_df = ec.compute_feature_importance(model_info)
        self.assertEqual(set(importance_df["feature"]), set(ec.FEATURE_COLUMNS))
        abs_coefs = importance_df["abs_coefficient"].tolist()
        self.assertEqual(abs_coefs, sorted(abs_coefs, reverse=True))
        # damage_line_density is the only feature that differs between classes here
        self.assertEqual(importance_df.iloc[0]["feature"], "damage_line_density")
        self.assertEqual(importance_df.iloc[0]["direction"], "toward_damaged")


class TestConfidenceStrata(unittest.TestCase):
    def _cv_eval_df(self, n):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.5, 1.0, size=n)
        return pd.DataFrame({
            "image_id": [f"img{i}" for i in range(n)],
            "true_label": ["clean"] * n,
            "cv_predicted_label": ["clean"] * n,
            "cv_confidence": conf,
            "cv_correct": [True] * n,
        })

    def test_strata_cover_every_row_exactly_once(self):
        for n in (3, 7, 25, 46):
            with self.subTest(n=n):
                df = self._cv_eval_df(n)
                parts = ec._partition_by_confidence(df)
                total = sum(len(p) for p in parts.values())
                self.assertEqual(total, n)
                all_ids = sorted(sum((p["image_id"].tolist() for p in parts.values()), []))
                self.assertEqual(all_ids, sorted(df["image_id"].tolist()))

    def test_all_three_strata_present_even_on_tiny_dataset(self):
        df = self._cv_eval_df(3)
        strata_df = ec.confidence_strata(df)
        self.assertEqual(set(strata_df["stratum"]), {"top20", "middle60", "bottom20"})
        self.assertEqual(strata_df["n"].sum(), 3)


class TestBuildSummary(unittest.TestCase):
    def test_summary_has_required_fields(self):
        rows = []
        for i in range(10):
            rows.append({"image_id": f"clean{i}", "true_label": "clean", **_features(damage_density=0.001)})
        for i in range(10):
            rows.append({"image_id": f"dmg{i}", "true_label": "damaged", **_features(damage_density=0.5)})
        table = pd.DataFrame(rows)

        cv_eval_df = ec.cross_validate(table)
        model_info = ec.fit_final_model(table)
        importance_df = ec.compute_feature_importance(model_info)
        strata_df = ec.confidence_strata(cv_eval_df)
        summary = ec.build_summary(cv_eval_df, importance_df, strata_df)

        for key in ("n_total", "n_clean_true", "n_damaged_true", "accuracy", "accuracy_95ci_wilson",
                    "confusion_matrix", "precision_recall_f1", "naive_baselines", "top_features",
                    "confidence_strata", "note", "feature_importance_note"):
            self.assertIn(key, summary)
        self.assertIn("clean", summary["precision_recall_f1"])
        self.assertIn("damaged", summary["precision_recall_f1"])
        self.assertEqual(len(summary["top_features"]), 5)
        self.assertEqual(summary["n_total"], 20)


class TestWilsonConfidenceInterval(unittest.TestCase):
    def test_matches_known_value_for_103_of_119(self):
        """Cross-checked against an independent manual computation (scipy
        norm.ppf(0.975) for z, plugged into the standard Wilson formula) -
        not just "does it run", but "is the number right": 103/119 correct
        should give roughly [0.793, 0.916], not the much wider/narrower
        interval a Wald or a bug would produce."""
        low, high = ec._wilson_ci(103, 119)
        self.assertAlmostEqual(low, 0.7927, places=3)
        self.assertAlmostEqual(high, 0.9155, places=3)

    def test_interval_widens_as_n_shrinks_for_the_same_proportion(self):
        low_big, high_big = ec._wilson_ci(87, 100)
        low_small, high_small = ec._wilson_ci(9, 10)  # ~same proportion, tiny n rooms it up
        self.assertGreater(high_small - low_small, high_big - low_big)

    def test_perfect_score_interval_stays_within_zero_one(self):
        low, high = ec._wilson_ci(20, 20)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_zero_total_returns_none(self):
        self.assertEqual(ec._wilson_ci(0, 0), (None, None))


class TestSupervisedBenchmarkEndToEnd(unittest.TestCase):
    """Runs run_supervised_benchmark() against a tiny fake feature table
    written to real feature_summary_json files on disk, exactly as
    load_feature_table() expects to read them from condition_analysis's
    output - the "tiny fake feature table" end-to-end case."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="fake_features_")
        rows = []
        gt_rows = []
        for i in range(6):
            image_id = f"clean{i}"
            path = os.path.join(cls.tmpdir, f"{image_id}_features.json")
            with open(path, "w") as f:
                json.dump(_features(damage_density=0.001, dirt=0.01), f)
            rows.append({"image_id": image_id, "visual_analysis_status": "ok", "feature_summary_json": path})
            gt_rows.append({"image_id": image_id, "true_label": "clean"})
        for i in range(6):
            image_id = f"dmg{i}"
            path = os.path.join(cls.tmpdir, f"{image_id}_features.json")
            with open(path, "w") as f:
                json.dump(_features(damage_density=0.5, dirt=0.01), f)
            rows.append({"image_id": image_id, "visual_analysis_status": "ok", "feature_summary_json": path})
            gt_rows.append({"image_id": image_id, "true_label": "damaged"})
        # one unusable row - must be excluded, not crash anything
        rows.append({"image_id": "broken0", "visual_analysis_status": "unusable", "feature_summary_json": None})
        gt_rows.append({"image_id": "broken0", "true_label": "damaged"})

        cls.condition_df = pd.DataFrame(rows)
        cls.ground_truth_df = pd.DataFrame(gt_rows)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_runs_end_to_end_on_tiny_fake_dataset(self):
        result = ec.run_supervised_benchmark(self.condition_df, self.ground_truth_df)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["cv_eval_df"]), 12)  # excludes the "unusable" row
        self.assertEqual(len(result["final_predictions_df"]), 12)
        self.assertEqual(len(result["importance_df"]), len(ec.FEATURE_COLUMNS))
        self.assertEqual(len(result["strata_df"]), 3)

    def test_merge_into_results_adds_expected_columns(self):
        result = ec.run_supervised_benchmark(self.condition_df, self.ground_truth_df)
        export_df = pd.DataFrame({"image_id": self.condition_df["image_id"], "condition": "uncertain"})
        merged = ec.merge_into_results(export_df, result["cv_eval_df"], result["final_predictions_df"])
        for col in ("external_predicted_label", "external_confidence",
                    "external_cv_predicted_label", "external_cv_correct"):
            self.assertIn(col, merged.columns)
        self.assertEqual(len(merged), len(export_df))

    def test_returns_none_when_a_class_has_too_few_samples(self):
        tiny_condition_df = self.condition_df[
            self.condition_df["image_id"].isin(["clean0", "dmg0", "dmg1", "dmg2"])
        ]
        tiny_gt_df = self.ground_truth_df[
            self.ground_truth_df["image_id"].isin(["clean0", "dmg0", "dmg1", "dmg2"])
        ]
        result = ec.run_supervised_benchmark(tiny_condition_df, tiny_gt_df)
        self.assertIsNone(result)


class TestNoLabelLeakageIntoNormalPipeline(unittest.TestCase):
    def test_condition_analysis_never_imports_external_classifier(self):
        """Structural guarantee: the label-free inference path
        (condition_analysis.py) has no import dependency on
        external_classifier.py - it's only named in an explanatory
        comment, never imported or called."""
        import_lines = "\n".join(
            line for line in inspect.getsource(condition_analysis).splitlines()
            if line.strip().startswith(("import ", "from "))
        )
        self.assertNotIn("external_classifier", import_lines)

    def test_self_evaluate_only_reads_synthetic_ground_truth_not_external(self):
        """self_evaluate's own "true_label" is the SYNTHETIC simulation's
        internal ground truth (config.SIM_GROUND_TRUTH_PATH) - a
        completely separate concept from the external folder label. This
        pins down that it never points at the external ground truth file
        instead."""
        source = inspect.getsource(condition_analysis.self_evaluate)
        self.assertIn("SIM_GROUND_TRUTH_PATH", source)
        self.assertNotIn("EXTERNAL_GROUND_TRUTH", source)

    def test_load_feature_table_is_the_only_place_labels_and_features_meet(self):
        source = inspect.getsource(ec.load_feature_table)
        self.assertIn("true_label", source)
        # everything else in the module operates on already-joined tables,
        # not on raw ground truth files
        for fn in (ec.cross_validate, ec.fit_final_model, ec.compute_feature_importance):
            self.assertNotIn("ground_truth", inspect.getsource(fn))


if __name__ == "__main__":
    unittest.main()

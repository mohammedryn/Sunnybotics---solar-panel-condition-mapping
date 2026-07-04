"""
Tests for the --external pipeline mode: CLI mode parsing, mode-directory
isolation, and the external dataset adapter processing a tiny, fully
synthetic clean/damaged fixture (no network, no dependency on the real
Sunnybotics dataset ever being cloned).

Run with: python3 -m unittest tests.test_external_dataset -v
"""
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import associate_panels, condition_analysis, config, export, external_dataset, ingest, priority_score
from scripts import run_pipeline


def _make_realistic_fixture_image(path, h, w, rng, crack=False):
    """Pure random noise gets smoothed to near-flat by the pipeline's
    area-averaging resize step (confirmed empirically during development
    - not a pipeline bug, just an unrealistic fixture), so this builds
    blocky-but-structured images that survive resizing with real
    variance, the way an actual photo would."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, h // 10):
        for c in range(0, w, w // 6):
            color = rng.integers(20, 70, size=3)
            img[r:r + h // 10, c:c + w // 6] = color
    if crack:
        cv2.line(img, (int(w * 0.1), int(h * 0.1)), (int(w * 0.8), int(h * 0.8)), (5, 5, 5), 4)
    cv2.imwrite(path, img)


class TestCliModeParsing(unittest.TestCase):
    def test_default_mode_is_synthetic(self):
        self.assertEqual(run_pipeline.parse_mode([]), "synthetic")

    def test_explicit_synthetic_flag(self):
        self.assertEqual(run_pipeline.parse_mode(["--synthetic"]), "synthetic")

    def test_explicit_external_flag(self):
        self.assertEqual(run_pipeline.parse_mode(["--external"]), "external")

    def test_both_flags_is_an_error(self):
        with self.assertRaises(SystemExit):
            run_pipeline.parse_mode(["--synthetic", "--external"])


class TestModeDirectoryIsolation(unittest.TestCase):
    def test_clean_mode_dirs_only_clears_active_mode(self):
        config.set_mode("synthetic")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(os.path.join(config.DATA_DIR, "sentinel_synthetic.txt"), "w") as f:
            f.write("synthetic")

        config.set_mode("external")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(os.path.join(config.DATA_DIR, "sentinel_external.txt"), "w") as f:
            f.write("external")

        run_pipeline._clean_mode_dirs()  # active mode is "external"

        self.assertFalse(os.path.exists(os.path.join("data/external", "sentinel_external.txt")))
        self.assertTrue(os.path.exists(os.path.join("data/synthetic", "sentinel_synthetic.txt")))

        shutil.rmtree("data/synthetic", ignore_errors=True)
        config.set_mode("synthetic")

    def test_clean_mode_dirs_preserves_cloned_external_dataset(self):
        """Regression test: config.EXTERNAL_DATASET_DIR lives INSIDE
        config.DATA_DIR for external mode (data/external/sunnybotics-...
        is a child of data/external). A naive rmtree(DATA_DIR) would
        delete the user's manually-cloned dataset on every run, forcing a
        re-clone each time - caught by actually running --external against
        the real cloned dataset, not just this fake fixture."""
        config.set_mode("external")
        os.makedirs(config.EXTERNAL_DATASET_DIR, exist_ok=True)
        with open(os.path.join(config.EXTERNAL_DATASET_DIR, "sentinel_dataset.txt"), "w") as f:
            f.write("do not delete")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(os.path.join(config.DATA_DIR, "captures_raw.csv"), "w") as f:
            f.write("stale generated file")

        run_pipeline._clean_mode_dirs()

        self.assertTrue(os.path.exists(os.path.join(config.EXTERNAL_DATASET_DIR, "sentinel_dataset.txt")))
        self.assertFalse(os.path.exists(os.path.join(config.DATA_DIR, "captures_raw.csv")))

        shutil.rmtree("data/external", ignore_errors=True)
        config.set_mode("synthetic")


class TestExternalDatasetAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="fake_external_dataset_")
        os.makedirs(os.path.join(cls.tmpdir, "clean"))
        os.makedirs(os.path.join(cls.tmpdir, "damaged"))
        rng = np.random.default_rng(7)
        for i in range(4):
            _make_realistic_fixture_image(os.path.join(cls.tmpdir, "clean", f"clean_{i}.jpg"), 800, 600, rng)
        for i in range(3):
            _make_realistic_fixture_image(
                os.path.join(cls.tmpdir, "damaged", f"damaged_{i}.jpg"), 1024, 768, rng, crack=True
            )

        config.set_mode("external")
        config.EXTERNAL_DATASET_DIR = cls.tmpdir
        shutil.rmtree(config.DATA_DIR, ignore_errors=True)
        shutil.rmtree(config.OUTPUTS_DIR, ignore_errors=True)

        cls.captures_df, cls.ground_truth_df = external_dataset.build_external_captures()
        cls.ingested = ingest.ingest()
        ingest.save_ingested(cls.ingested)
        cls.associated = associate_panels.associate()
        associate_panels.save_association(cls.associated)
        cls.conditioned = condition_analysis.analyze()
        condition_analysis.save_results(cls.conditioned)
        cls.prioritized = priority_score.score_priorities()
        priority_score.save_priorities(cls.prioritized)
        cls.export_df = export.run_export()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        shutil.rmtree(config.DATA_DIR, ignore_errors=True)
        shutil.rmtree(config.OUTPUTS_DIR, ignore_errors=True)
        config.set_mode("synthetic")

    def test_missing_dataset_raises_helpful_error_not_silent_failure(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(external_dataset.ExternalDatasetNotFoundError) as ctx:
                external_dataset.discover_images(empty_dir)
            self.assertIn("git clone", str(ctx.exception))
            self.assertIn(config.EXTERNAL_DATASET_CLONE_URL, str(ctx.exception))

    def test_all_seven_images_discovered(self):
        self.assertEqual(len(self.captures_df), 7)
        self.assertEqual(len(self.ground_truth_df), 7)

    def test_folder_label_not_present_in_captures_raw(self):
        """The whole point of keeping ground truth separate: captures_raw
        (what ingestion/association/condition-analysis/priority actually
        read) must not contain the folder label in any column, only the
        completely separate ground_truth_df does."""
        self.assertNotIn("true_label", self.captures_df.columns)
        self.assertNotIn("label", self.captures_df.columns)
        self.assertIn("true_label", self.ground_truth_df.columns)

    def test_no_panel_identity_in_external_captures_either(self):
        """Same invariant test_pipeline.py checks for synthetic mode -
        applies here too, since RF-06 doesn't apply to this dataset at all."""
        self.assertNotIn("panel_id", self.captures_df.columns)
        self.assertNotIn("panel_row", self.captures_df.columns)

    def test_pipeline_does_not_crash_and_association_is_unresolvable(self):
        self.assertEqual(len(self.associated), 7)
        self.assertTrue((self.associated["association_status"] == "unresolvable").all())

    def test_final_export_has_required_e2_fields(self):
        required = {
            "image_id", "timestamp", "latitude", "longitude", "robot_id", "mission_id",
            "panel_row", "panel_id", "condition", "confidence", "cleaning_priority_score",
            "annotated_image_path", "detected_issues",
        }
        self.assertTrue(required.issubset(set(self.export_df.columns)))
        self.assertEqual(len(self.export_df), 7)

    def test_panel_identity_is_honestly_null_not_fabricated(self):
        self.assertTrue(self.export_df["panel_id"].isna().all())
        self.assertTrue(self.export_df["latitude"].isna().all())

    def test_eval_summary_joins_label_only_after_inference(self):
        """The label is read here, at evaluation time, for the first and
        only time in the whole external-mode run - never during
        ingest/associate/condition_analysis/priority_score above."""
        summary = external_dataset.external_eval_summary(self.export_df)
        self.assertEqual(summary["n_total"], 7)
        self.assertIn("confusion_matrix", summary)
        self.assertIn("note", summary)


if __name__ == "__main__":
    unittest.main()

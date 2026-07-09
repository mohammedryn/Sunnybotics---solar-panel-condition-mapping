import unittest
import pandas as pd
from src import priority_score as ps


class TestScoreExternalRow(unittest.TestCase):
    def test_damaged_floors_high_and_scales_with_confidence(self):
        row = pd.Series({
            "image_id": "x", "condition": "damaged", "condition_confidence": 0.9,
            "visual_analysis_status": "ok",
        })
        result = ps._score_external_row(row)
        self.assertEqual(result["recommended_action"], "inspect")
        self.assertAlmostEqual(result["cleaning_priority_score"], 70.0 + 30.0 * 0.9)
        self.assertEqual(result["priority_band"], "critical")

    def test_clean_scores_low(self):
        row = pd.Series({
            "image_id": "x", "condition": "clean", "condition_confidence": 0.95,
            "visual_analysis_status": "ok",
        })
        result = ps._score_external_row(row)
        self.assertEqual(result["recommended_action"], "none")
        self.assertEqual(result["cleaning_priority_score"], 5.0)

    def test_unusable_row_recaptures_regardless_of_condition(self):
        row = pd.Series({
            "image_id": "x", "condition": "uncertain", "condition_confidence": None,
            "visual_analysis_status": "unusable", "unusable_reason": "blurry",
        })
        result = ps._score_external_row(row)
        self.assertEqual(result["recommended_action"], "recapture")
        self.assertEqual(result["cleaning_priority_score"], 70.0)

    def test_score_priorities_external_mode_uses_external_scoring(self):
        # score_priorities(mode="external") must not touch detected_issues
        # at all - a row with "damage" in detected_issues but
        # condition="clean" (the exact scenario proven to occur on 100%
        # of real rows) must score as clean, not damaged.
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "condition_results.csv")
            pd.DataFrame({
                "image_id": ["a"], "condition": ["clean"], "condition_confidence": [0.8],
                "visual_analysis_status": ["ok"], "detected_issues": ["dirt,shadow,damage"],
            }).to_csv(path, index=False)
            result = ps.score_priorities(condition_results_path=path, mode="external")
            self.assertEqual(result.iloc[0]["recommended_action"], "none")

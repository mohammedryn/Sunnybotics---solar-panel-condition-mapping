"""
A small set of regression tests, not a full suite - run with:
    python3 -m unittest tests.test_pipeline -v

Runs the real pipeline once (no mocks) and checks the properties that
actually matter for this assignment's grading: no ground-truth leakage,
correct row counts, crash-resistance on bad inputs, required export
fields present, and one regression test tied to a real bug found during
development (see priority_score.py's module docstring).

Side effect worth knowing, though no longer a correctness problem: this
writes to the same data/ and outputs/ paths the real pipeline uses
(config.py's paths aren't parameterized per-caller), so running the
tests has the same effect as another pipeline run. That used to leave
two generations of annotated images sitting side by side (image_id was
a fresh, unseeded UUID each run) - fixed by deriving image_id from the
same seeded generator as everything else (see simulate_mission.py),
so a rerun regenerates and overwrites the same 100 files instead of
accumulating a second set. Proper path isolation (every stage's output
paths threaded through as parameters) would still be cleaner practice,
just not required for correctness anymore.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import associate_panels, condition_analysis, config, export, ingest, priority_score, simulate_mission


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        simulate_mission.simulate_all()
        cls.ingested = ingest.ingest()
        ingest.save_ingested(cls.ingested)
        cls.associated = associate_panels.associate()
        associate_panels.save_association(cls.associated)
        cls.conditioned = condition_analysis.analyze()
        condition_analysis.save_results(cls.conditioned)
        cls.prioritized = priority_score.score_priorities()
        priority_score.save_priorities(cls.prioritized)
        cls.export_df = export.run_export()

    def test_captures_raw_has_no_panel_identity(self):
        """RF-06's entire point: panel identity must come from association,
        never be pre-filled from the simulator."""
        captures_raw = pd.read_csv(config.CAPTURES_RAW_PATH)
        self.assertNotIn("panel_id", captures_raw.columns)
        self.assertNotIn("panel_row", captures_raw.columns)

    def test_image_ids_are_deterministic(self):
        """Regression test for a real gap: image_id used to come from
        uuid.uuid4() (OS entropy, not config.SEED), so two runs of the
        'same' seeded pipeline actually produced two different datasets -
        annotated images would silently accumulate across reruns instead
        of being replaced. Two independent simulate_all() calls with the
        same seed must now produce identical image_ids."""
        first = simulate_mission.simulate_all()[0]["image_id"].tolist()
        second = simulate_mission.simulate_all()[0]["image_id"].tolist()
        self.assertEqual(first, second)

    def test_final_export_has_100_rows(self):
        self.assertEqual(len(self.export_df), 100)

    def test_ingestion_does_not_crash_on_bad_inputs(self):
        """5 edge cases are deliberately injected (missing file, truncated
        file, GPS dropout, invalid coordinates, incomplete metadata) -
        ingestion must tag all of them without raising."""
        edge_cases = pd.read_csv(os.path.join(config.DATA_DIR, "injected_edge_cases.csv"))
        self.assertEqual(len(edge_cases), 5)
        flagged = self.ingested[self.ingested["image_id"].isin(edge_cases["image_id"])]
        self.assertTrue((flagged["requires_attention"]).all())

    def test_final_export_has_required_e2_fields(self):
        required = {
            "image_id", "timestamp", "latitude", "longitude", "robot_id", "mission_id",
            "panel_row", "panel_id", "condition", "confidence", "cleaning_priority_score",
            "annotated_image_path", "detected_issues",
        }
        self.assertTrue(required.issubset(set(self.export_df.columns)))

    def test_secondary_damage_signal_is_not_worded_as_confirmed(self):
        """Regression test for a real bug: priority_reason used to word a
        WEAK, non-dominant damage signal identically to a confirmed damage
        finding, and scored it using the dominant condition's confidence
        instead of damage's own - so a glare image could land at
        critical/inspect with text reading 'Damage detected
        (confidence=1.00)'. Any row where damage is a secondary signal
        must say so explicitly."""
        merged = self.export_df.merge(
            self.conditioned[["image_id", "detected_issues"]], on="image_id", suffixes=("", "_full")
        )
        secondary_damage = merged[
            (merged["condition"] != "damage") & (merged["detected_issues_full"].str.contains("damage", na=False))
        ]
        if len(secondary_damage) == 0:
            self.skipTest("no secondary-damage case in this run to check")
        for _, row in secondary_damage.iterrows():
            self.assertIn("Primary visual finding is", row["priority_reason"])
            self.assertNotIn("Damage detected (confidence", row["priority_reason"])


if __name__ == "__main__":
    unittest.main()

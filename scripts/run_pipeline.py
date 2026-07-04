#!/usr/bin/env python3
"""
Runs the pipeline end-to-end, one command, in one of two modes:

  --synthetic (default) - the full simulated mission dataset: GPS, route,
      odometry, panel association, all 5 condition categories, injected
      edge cases. This is the main, full end-to-end validation.

  --external - Sunnybotics' real sample dataset (clean/ and damaged/
      image folders only, no GPS/route/panel metadata). A supplemental
      sanity check for the visual-analysis stage specifically, not a
      replacement for --synthetic - see src/external_dataset.py.

Usage:
  python3 scripts/run_pipeline.py                # synthetic (default)
  python3 scripts/run_pipeline.py --synthetic
  python3 scripts/run_pipeline.py --external

Each mode's data/ and outputs/ are cleared and regenerated fresh on every
run (data/<mode>/, outputs/<mode>/) - running one mode never touches the
other mode's results.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    annotate, associate_panels, condition_analysis, config, export,
    external_classifier, external_dataset, ingest, priority_score,
    simulate_mission, visualize,
)


def parse_mode(argv=None) -> str:
    """argv defaults to sys.argv[1:] (real CLI use); tests pass an
    explicit list so mode-parsing is checkable without spawning a
    subprocess or running the actual pipeline."""
    parser = argparse.ArgumentParser(description="Run the Sunnybotics panel-condition pipeline.")
    parser.add_argument("--synthetic", action="store_true", help="Simulated mission dataset (default).")
    parser.add_argument("--external", action="store_true", help="Sunnybotics' real sample dataset (clean/damaged folders).")
    args = parser.parse_args(argv)
    if args.synthetic and args.external:
        parser.error("--synthetic and --external are mutually exclusive - pick one.")
    return "external" if args.external else "synthetic"


def _clean_mode_dirs():
    """Clears ONLY the active mode's own directories - config.set_mode()
    must run before this so config.DATA_DIR/OUTPUTS_DIR point at the
    right mode. Never touches the other mode's data/outputs.

    External mode is special: config.EXTERNAL_DATASET_DIR (the user's
    manually-cloned dataset) lives INSIDE config.DATA_DIR
    ("data/external/sunnybotics-solar-panel-challenge" under
    "data/external"), not alongside it. A plain rmtree of DATA_DIR would
    delete the cloned dataset on every single run, forcing a re-clone each
    time - so for external mode this clears every sibling of the dataset
    folder instead of the whole tree."""
    shutil.rmtree(config.OUTPUTS_DIR, ignore_errors=True)

    if config.MODE == "external" and os.path.isdir(config.DATA_DIR):
        preserve = os.path.abspath(config.EXTERNAL_DATASET_DIR)
        for entry in os.listdir(config.DATA_DIR):
            full = os.path.abspath(os.path.join(config.DATA_DIR, entry))
            if full == preserve:
                continue
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.remove(full)
    else:
        shutil.rmtree(config.DATA_DIR, ignore_errors=True)


def _acquire_data(mode: str):
    if mode == "synthetic":
        print("\n" + "=" * 60)
        print("Simulating missions (RF-01/RF-02)")
        print("=" * 60)
        simulate_mission.simulate_all()
        return

    print("\n" + "=" * 60)
    print("Loading external dataset (Sunnybotics sample: clean/ + damaged/)")
    print("=" * 60)
    try:
        captures_df, _ = external_dataset.build_external_captures()
    except external_dataset.ExternalDatasetNotFoundError as e:
        print(f"\n{e}\n")
        sys.exit(1)
    print(f"Loaded {len(captures_df)} external images.")


def _run_shared_stages(mode: str):
    """Ingestion through export/visualization - identical code path for
    both modes. ingest.py, associate_panels.py, condition_analysis.py,
    priority_score.py, annotate.py, and export.py are all schema-tolerant
    of external mode's missing route/GPS structure (each documents this
    in its own module docstring) - nothing here branches on mode except
    which self-evaluation/visualization to run at the end, since
    synthetic's ground truth and external's folder-label ground truth
    have genuinely different shapes."""
    print("\n" + "=" * 60)
    print("Ingesting captures (RF-01)")
    print("=" * 60)
    ingested = ingest.ingest()
    ingest.save_ingested(ingested)
    print(f"Ingested {len(ingested)} rows.")

    print("\n" + "=" * 60)
    print("Associating panels (RF-06)")
    print("=" * 60)
    associated = associate_panels.associate()
    associate_panels.save_association(associated)
    print(f"Associated {len(associated)} rows.")
    if mode == "synthetic":
        for k, v in associate_panels.self_evaluate(associated).items():
            print(f"  {k}: {v}")
    else:
        print("  spatial association: not applicable (no GPS/route data in this dataset)")

    print("\n" + "=" * 60)
    print("Analyzing condition (RF-03)")
    print("=" * 60)
    conditioned = condition_analysis.analyze()
    condition_analysis.save_results(conditioned)
    print(f"Analyzed {len(conditioned)} rows.")
    if mode == "synthetic":
        for k, v in condition_analysis.self_evaluate(conditioned).items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Scoring priority (RF-04)")
    print("=" * 60)
    prioritized = priority_score.score_priorities()
    priority_score.save_priorities(prioritized)
    print(f"Scored {len(prioritized)} rows.")

    print("\n" + "=" * 60)
    print("Annotating + exporting + visualizing (RF-05/RF-07)")
    print("=" * 60)
    export_df = export.run_export()
    print(f"Exported {len(export_df)} rows -> {config.OUTPUTS_DIR}/results.{{json,csv,geojson}}")
    print(f"Annotated images -> {config.ANNOTATED_DIR}/")

    if mode == "synthetic":
        visualize.static_farm_grid(export_df)
        visualize.interactive_farm_grid(export_df)
        print(f"Farm grid -> {config.VISUALIZATIONS_DIR}/farm_grid.png, "
              f"{config.VISUALIZATIONS_DIR}/farm_grid_interactive.html")
    else:
        chart_path = external_dataset.build_external_summary_chart(export_df)
        zero_shot_summary = external_dataset.external_eval_summary(export_df)
        external_dataset.save_external_eval_summary(zero_shot_summary)
        print(f"Summary chart -> {chart_path}")
        print(f"Zero-shot rule classifier eval -> {config.OUTPUTS_DIR}/external_eval_summary.json")
        print(f"  clean_vs_damaged_accuracy_on_binary_predictions (zero-shot, synthetic-calibrated rules): "
              f"{zero_shot_summary['clean_vs_damaged_accuracy_on_binary_predictions']}")
        print(f"  n_evaluable: {zero_shot_summary['n_evaluable']} / n_total: {zero_shot_summary['n_total']}")

        print("\n" + "=" * 60)
        print("Supervised clean/damaged benchmark (cross-validated) (RF-03 supplement)")
        print("=" * 60)
        ground_truth_df = pd.read_csv(config.EXTERNAL_GROUND_TRUTH_PATH)
        benchmark = external_classifier.run_supervised_benchmark(conditioned, ground_truth_df)
        if benchmark is None:
            print("  Not enough labeled images per class to run a meaningful benchmark - skipped.")
        else:
            external_classifier.save_binary_eval(benchmark["cv_eval_df"])
            external_classifier.save_binary_eval_summary(benchmark["summary"])
            external_classifier.save_feature_importance(benchmark["importance_df"])
            external_classifier.save_confidence_strata(benchmark["strata_df"])

            export_df = external_classifier.merge_into_results(
                export_df, benchmark["cv_eval_df"], benchmark["final_predictions_df"]
            )
            export.save_json(export_df)
            export.save_csv(export_df)
            export.save_geojson(export_df)

            summary = benchmark["summary"]
            print(f"  classifier backend: {summary['classifier_backend']} ({summary['cv_folds']}-fold CV)")
            ci = summary["accuracy_95ci_wilson"]
            print(f"  cross-validated accuracy (main external metric): {summary['accuracy']} "
                  f"(95% CI [Wilson]: {ci[0]:.3f}-{ci[1]:.3f})")
            print(f"  naive baselines: {summary['naive_baselines']}")
            print(f"  top features: {[f['feature'] for f in summary['top_features']]}")
            print(f"  -> {config.OUTPUTS_DIR}/external_binary_eval.csv, "
                  f"external_binary_eval_summary.json, external_feature_importance.csv, "
                  f"external_confidence_strata.csv")

    return export_df


def main():
    mode = parse_mode()
    config.set_mode(mode)
    _clean_mode_dirs()

    print("=" * 60)
    print(f"Mode: {mode}")
    print("=" * 60)

    _acquire_data(mode)
    _run_shared_stages(mode)

    print("\nDone.")


if __name__ == "__main__":
    main()

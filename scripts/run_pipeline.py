#!/usr/bin/env python3
"""
Runs the entire pipeline end-to-end, one command, per the brief's
"1 or 2 commands documented in the README" requirement:

  simulate -> ingest -> associate -> condition analysis -> priority score
  -> annotate + export (results.json/.csv/.geojson) + farm grid visualizations

Usage: python3 scripts/run_pipeline.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulate_mission, ingest, associate_panels, condition_analysis, priority_score, export, visualize


def main():
    print("=" * 60)
    print("1/6 Simulating missions (RF-01/RF-02)")
    print("=" * 60)
    simulate_mission.simulate_all()

    print("\n" + "=" * 60)
    print("2/6 Ingesting captures (RF-01)")
    print("=" * 60)
    ingested = ingest.ingest()
    ingest.save_ingested(ingested)
    print(f"Ingested {len(ingested)} rows.")

    print("\n" + "=" * 60)
    print("3/6 Associating panels (RF-06)")
    print("=" * 60)
    associated = associate_panels.associate()
    associate_panels.save_association(associated)
    print(f"Associated {len(associated)} rows.")
    for k, v in associate_panels.self_evaluate(associated).items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("4/6 Analyzing condition (RF-03)")
    print("=" * 60)
    conditioned = condition_analysis.analyze()
    condition_analysis.save_results(conditioned)
    print(f"Analyzed {len(conditioned)} rows.")
    for k, v in condition_analysis.self_evaluate(conditioned).items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("5/6 Scoring priority (RF-04)")
    print("=" * 60)
    prioritized = priority_score.score_priorities()
    priority_score.save_priorities(prioritized)
    print(f"Scored {len(prioritized)} rows.")

    print("\n" + "=" * 60)
    print("6/6 Annotating + exporting + visualizing (RF-05/RF-07)")
    print("=" * 60)
    export_df = export.run_export()
    print(f"Exported {len(export_df)} rows -> outputs/results.{{json,csv,geojson}}")
    print(f"Annotated images -> {export.config.ANNOTATED_DIR}/")

    visualize.static_farm_grid(export_df)
    visualize.interactive_farm_grid(export_df)
    print("Farm grid -> outputs/visualizations/farm_grid.png, outputs/visualizations/farm_grid_interactive.html")

    print("\nDone.")


if __name__ == "__main__":
    main()

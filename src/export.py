"""
RF-07: final export - joins the full traceability chain into one row per
image and writes results.json / results.csv / results.geojson.

Traceability chain (each stage's own module owns its own logic; this
module only joins their outputs on image_id, it re-derives nothing):
  raw capture (captures_raw.csv)
  -> ingestion status (ingested_captures.csv, RF-01)
  -> association result (association_results.csv, RF-06)
  -> condition result (condition_results.csv, RF-03)
  -> priority decision (priority_results.csv, RF-04)
  -> annotated image (outputs/annotated/, RF-05)
  -> this export (RF-07)

Field mapping worth being explicit about: `panel_row`/`panel_id` come
from association's predicted_panel_row/predicted_panel_id - RF-06's
output, never ground truth, since resolving that identity from noisy
GPS+odometry is the entire point of RF-06. `latitude`/`longitude` stay
as the raw GPS-reported reading (RF-02's actual sensor value), kept
visibly distinct from the resolved panel position - "where the robot's
GPS said it was" and "which panel we believe this is" are different
claims with different confidence, and collapsing them would erase
exactly the traceability this project has been built around.
`confidence` is renamed from condition_confidence to match the brief's
literal E2 field name.

Every image_id that entered ingestion appears in the final export, even
if unresolvable/unusable - dropping rows silently would break the "every
image answers where/what/evidence" requirement for exactly the images
where that answer is "we don't know, and here's why."
"""
import json
import os

import pandas as pd

from . import annotate, config


def build_joined_dataframe(
    ingested_captures_path: str = None,
    association_results_path: str = None,
    condition_results_path: str = None,
    priority_results_path: str = None,
) -> pd.DataFrame:
    ingested_captures_path = ingested_captures_path or os.path.join(config.DATA_DIR, "ingested_captures.csv")
    association_results_path = association_results_path or os.path.join(config.DATA_DIR, "association_results.csv")
    condition_results_path = condition_results_path or os.path.join(config.DATA_DIR, "condition_results.csv")
    priority_results_path = priority_results_path or os.path.join(config.DATA_DIR, "priority_results.csv")

    captures = pd.read_csv(ingested_captures_path)
    association = pd.read_csv(association_results_path)
    condition = pd.read_csv(condition_results_path)
    priority = pd.read_csv(priority_results_path)

    df = captures.merge(association, on="image_id", how="left", suffixes=("", "_assoc"))
    df = df.merge(condition, on="image_id", how="left", suffixes=("", "_cond"))
    df = df.merge(priority, on="image_id", how="left", suffixes=("", "_prio"))

    # Recover mission_id from route_pass_id ("{mission_id}-{nominal_row_id}")
    # when missing - caught via the farm grid visualization silently
    # dropping a row (the deliberately-injected incomplete-metadata edge
    # case from Section 1): mission_id was blanked to test RF-01's
    # handling, but route_pass_id was untouched, so the mission is still
    # recoverable. This does NOT erase the original gap - it remains
    # visible in ingested_captures.csv's metadata_status for anyone
    # auditing RF-01 specifically; this recovery only affects the final
    # delivered export, which should be maximally useful rather than
    # silently propagating a gap we have the means to close.
    # route_pass_id only exists for datasets with real route structure
    # (synthetic mode) - the external dataset has no such column at all,
    # which is a different situation than "this specific row is missing
    # it" and shouldn't be treated as the same recovery case.
    if "route_pass_id" in df.columns:
        missing_mission = df["mission_id"].isna() & df["route_pass_id"].notna()
        df.loc[missing_mission, "mission_id"] = df.loc[missing_mission, "route_pass_id"].str.split("-").str[0]
    return df


def build_export_rows(joined_df: pd.DataFrame, annotated_paths: dict) -> pd.DataFrame:
    export = pd.DataFrame({
        "image_id": joined_df["image_id"],
        "timestamp": joined_df["timestamp"],
        "latitude": joined_df["latitude"],
        "longitude": joined_df["longitude"],
        "robot_id": joined_df["robot_id"],
        "mission_id": joined_df["mission_id"],
        "panel_row": joined_df["predicted_panel_row"],
        "panel_id": joined_df["predicted_panel_id"],
        "condition": joined_df["condition"],
        "confidence": joined_df["condition_confidence"],
        "zero_shot_condition": joined_df.get("zero_shot_condition"),
        "zero_shot_confidence": joined_df.get("zero_shot_confidence"),
        "cleaning_priority_score": joined_df["cleaning_priority_score"],
        "annotated_image_path": joined_df["image_id"].map(annotated_paths),
        "detected_issues": joined_df["detected_issues"],
        "association_status": joined_df["association_status"],
        "association_confidence": joined_df["association_confidence"],
        "visual_analysis_status": joined_df["visual_analysis_status"],
        "priority_band": joined_df["priority_band"],
        "recommended_action": joined_df["recommended_action"],
        "priority_reason": joined_df["priority_reason"],
    })
    return export


def save_json(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", indent=2)


def save_csv(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.OUTPUTS_DIR, "results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_geojson(df: pd.DataFrame, path: str = None):
    """Point geometry at the raw GPS-reported position (not the resolved
    panel position - see module docstring)."""
    path = path or os.path.join(config.OUTPUTS_DIR, "results.geojson")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    features = []
    for _, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        geometry = None
        if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            geometry = {"type": "Point", "coordinates": [lon, lat]}
        properties = row.drop(["latitude", "longitude"]).to_dict()
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2, default=str)


def run_export():
    joined = build_joined_dataframe()
    annotated_paths = annotate.annotate_all(joined)
    export_df = build_export_rows(joined, annotated_paths)
    save_json(export_df)
    save_csv(export_df)
    save_geojson(export_df)
    return export_df


if __name__ == "__main__":
    df = run_export()
    print(f"Exported {len(df)} rows -> outputs/results.{{json,csv,geojson}}")
    print(f"Annotated images -> {config.ANNOTATED_DIR}/")
    print("\nrecommended_action:\n", df["recommended_action"].value_counts().to_string())
    print("\npanel_row (unresolved count):", df["panel_row"].isna().sum())

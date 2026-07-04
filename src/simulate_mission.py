"""
Orchestrates RF-01/RF-02: simulates two robot passes (missions) over the
farm and produces four artifacts, kept strictly separate so RF-06
(panel association) has something real to solve later:

  data/captures_raw.csv           - what a real robot would actually produce
                                     (image_id, timestamp, noisy GPS fix,
                                     robot_id, mission_id, route/odometry
                                     context, image path). NO confirmed panel
                                     identity here.
  data/route_plan.csv             - one row per route_pass_id: the robot's
                                     own navigation plan (which nominal row,
                                     which direction, expected panel count,
                                     planned start/end position). This is a
                                     real input a robot's path planner would
                                     have, independent of vision - but it is
                                     a PRIOR, not a confirmed identity. The
                                     robot can drift off-plan, so association
                                     (RF-06) still has to verify it against
                                     GPS+odometry evidence rather than trust
                                     it blindly.
  data/sim_ground_truth_internal.csv - true panel/condition labels, used only
                                     for later self-evaluation. Never fed to
                                     the association or condition-analysis
                                     algorithms.
  data/injected_edge_cases.csv    - a log of deliberately broken rows
                                     (missing file, truncated file, GPS
                                     dropout, invalid coordinates, incomplete
                                     identity metadata) used to exercise
                                     RF-01's crash-resistance
                                     requirement.

Persistent panel conditions (clean/dirty/damaged) are assigned once per
panel and carry across both missions. Transient overlays (shadow/glare)
are re-rolled per mission based on that mission's simulated sun
position. That split is what makes a dirt-vs-shadow discrimination
strategy (required report Question 3) actually checkable against data,
instead of just asserted.

Rows are traversed in alternating directions (boustrophedon: row 0
eastbound, row 1 westbound, row 2 eastbound, ...), which is how a real
cleaning/inspection robot would minimize repositioning time. This means
`odom_delta_m` alone is not enough to localize a capture - without
knowing which route_pass_id and direction a capture belongs to, rising
odometry could mean "moving toward panel 10" or "moving toward panel 1"
depending on the row. `route_pass_id`, `nominal_row_id`,
`route_direction`, `capture_seq_in_pass`, and `odom_cumulative_m`
(distance from route start, in the row's direction) are recorded
explicitly so the association step never has to infer this from row
order in the file.
"""
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, farm_layout, gps_model, image_synth, odometry_model


def _assign_persistent_conditions(farm_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    conditions = list(config.PERSISTENT_CONDITION_WEIGHTS.keys())
    weights = list(config.PERSISTENT_CONDITION_WEIGHTS.values())
    choices = rng.choice(conditions, size=len(farm_df), p=weights)
    return dict(zip(farm_df["panel_id"], choices))


def _assign_transient_overlay(mission_id: str, rng: np.random.Generator) -> str:
    weights_map = config.TRANSIENT_OVERLAY_WEIGHTS[mission_id]
    overlays = list(weights_map.keys())
    weights = list(weights_map.values())
    return rng.choice(overlays, p=weights)


def _inject_ingestion_edge_cases(captures_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                                  rng: np.random.Generator) -> pd.DataFrame:
    """Deliberately breaks 5 rows to exercise RF-01's 'must not crash on a
    missing/corrupted image, incomplete metadata, or invalid coordinates'
    requirement, each in a distinct, clearly-labeled way."""
    already_corrupted_ids = set(ground_truth_df.loc[ground_truth_df["true_is_corrupted"], "image_id"])
    candidates = captures_df.index[~captures_df["image_id"].isin(already_corrupted_ids)].tolist()
    idx_missing, idx_truncated, idx_dropout, idx_invalid, idx_incomplete = rng.choice(
        candidates, size=5, replace=False
    )

    missing_path = captures_df.loc[idx_missing, "image_path"]
    if os.path.exists(missing_path):
        os.remove(missing_path)

    trunc_path = captures_df.loc[idx_truncated, "image_path"]
    if os.path.exists(trunc_path):
        with open(trunc_path, "rb") as f:
            data = f.read()
        cut = int(len(data) * rng.uniform(0.2, 0.5))
        with open(trunc_path, "wb") as f:
            f.write(data[:cut])

    captures_df.loc[idx_dropout, ["latitude", "longitude"]] = np.nan
    captures_df.loc[idx_invalid, ["latitude", "longitude"]] = 999.0
    captures_df.loc[idx_incomplete, "mission_id"] = np.nan

    log = pd.DataFrame([
        {"image_id": captures_df.loc[idx_missing, "image_id"], "edge_case": "missing_image_file"},
        {"image_id": captures_df.loc[idx_truncated, "image_id"], "edge_case": "truncated_corrupted_file"},
        {"image_id": captures_df.loc[idx_dropout, "image_id"], "edge_case": "gps_dropout_nan_coords"},
        {"image_id": captures_df.loc[idx_invalid, "image_id"], "edge_case": "invalid_out_of_range_coords"},
        {"image_id": captures_df.loc[idx_incomplete, "image_id"], "edge_case": "incomplete_metadata_missing_mission_id"},
    ])
    log.to_csv(os.path.join(config.DATA_DIR, "injected_edge_cases.csv"), index=False)
    return captures_df


def simulate_all(seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    # Independent stream, spawned from the same seed, dedicated to
    # image_id generation. Deliberately NOT drawing from `rng` directly:
    # an earlier version did, and consuming 16 bytes per capture from the
    # shared stream shifted every subsequent draw (conditions, GPS noise,
    # which captures get corrupted) - meaning fixing image_id
    # determinism silently changed the whole simulated dataset as a side
    # effect. Spawning a separate reproducible stream decouples the two
    # concerns completely: touching ID generation can never again perturb
    # the substantive simulation randomness.
    id_rng = rng.spawn(1)[0]

    os.makedirs(config.RAW_IMAGES_DIR, exist_ok=True)

    farm_df = farm_layout.generate_farm()
    farm_layout.save_farm_truth(farm_df)
    persistent_conditions = _assign_persistent_conditions(farm_df, rng)

    total_captures = config.NUM_ROWS * config.PANELS_PER_ROW * len(config.MISSION_IDS)
    n_corrupted = max(1, int(total_captures * config.CORRUPTED_CAPTURE_RATE))
    corrupted_indices = set(rng.choice(total_captures, size=n_corrupted, replace=False).tolist())
    visual_corruption_modes = ["heavy_blur", "blackout", "overexposed"]

    captures, ground_truth, route_plan = [], [], []
    capture_counter = 0

    for mission_id in config.MISSION_IDS:
        mission_bias = gps_model.sample_mission_bias(rng)
        current_time = datetime.fromisoformat(config.MISSION_START_TIMES[mission_id])

        for row_idx in range(config.NUM_ROWS):
            nominal_row_id = f"R{row_idx + 1}"
            route_pass_id = f"{mission_id}-{nominal_row_id}"
            eastbound = (row_idx % 2 == 0)
            route_direction = "eastbound" if eastbound else "westbound"
            heading_rad = 0.0 if eastbound else np.pi
            panel_idx_sequence = range(config.PANELS_PER_ROW) if eastbound \
                else range(config.PANELS_PER_ROW - 1, -1, -1)

            odom_cumulative = 0.0
            start_panel = farm_df.iloc[row_idx * config.PANELS_PER_ROW + panel_idx_sequence[0]]
            end_panel = farm_df.iloc[row_idx * config.PANELS_PER_ROW + panel_idx_sequence[-1]]
            route_plan.append({
                "route_pass_id": route_pass_id,
                "mission_id": mission_id,
                "nominal_row_id": nominal_row_id,
                "start_east_m": start_panel["true_east_m"],
                "start_north_m": start_panel["true_north_m"],
                "end_east_m": end_panel["true_east_m"],
                "end_north_m": end_panel["true_north_m"],
                "route_direction": route_direction,
                "expected_panel_count": config.PANELS_PER_ROW,
            })

            for capture_seq, panel_idx in enumerate(panel_idx_sequence):
                panel = farm_df.iloc[row_idx * config.PANELS_PER_ROW + panel_idx]
                panel_id, panel_row = panel["panel_id"], panel["panel_row"]
                true_east, true_north = panel["true_east_m"], panel["true_north_m"]

                persistent_condition = persistent_conditions[panel_id]
                transient_overlay = _assign_transient_overlay(mission_id, rng)

                is_corrupted = capture_counter in corrupted_indices
                corruption_mode = visual_corruption_modes[capture_counter % 3] if is_corrupted else None

                # Deterministic, not uuid.uuid4(): that draws from OS
                # entropy, not config.SEED, so two runs produced two
                # different datasets under the same "fixed seed" claim -
                # a real gap, found via this project's own rescoring
                # (annotated images silently piling up across reruns
                # instead of being replaced). Deriving the UUID's bytes
                # from the same seeded generator as everything else makes
                # the entire dataset - image_id included - reproducible,
                # not just its statistics.
                image_id = str(uuid.UUID(bytes=id_rng.bytes(16), version=4))
                timestamp = current_time.isoformat()
                current_time += timedelta(seconds=config.CAPTURE_INTERVAL_S)

                gps_east, gps_north = gps_model.simulate_gps_fix(
                    true_east, true_north, mission_bias, rng, heading_rad=heading_rad
                )
                lat, lon = farm_layout.enu_to_latlon(gps_east, gps_north)

                # Along-track distance since the previous capture in THIS route_pass_id
                # (physical panel pitch, direction-agnostic magnitude).
                true_delta_m = 0.0 if capture_seq == 0 else config.PANEL_PITCH_M
                odom_delta = odometry_model.simulate_odometry_delta(true_delta_m, rng)
                odom_cumulative += odom_delta

                img, issues = image_synth.synthesize_capture_image(
                    rng, persistent_condition, transient_overlay, mission_id,
                    is_corrupted=is_corrupted, corruption_mode=corruption_mode,
                )
                image_path = os.path.join(config.RAW_IMAGES_DIR, f"{image_id}.jpg")
                image_synth.save_image(img, image_path)

                captures.append({
                    "image_id": image_id,
                    "timestamp": timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "robot_id": config.ROBOT_ID,
                    "mission_id": mission_id,
                    "route_pass_id": route_pass_id,
                    "nominal_row_id": nominal_row_id,
                    "route_direction": route_direction,
                    "capture_seq_in_pass": capture_seq,
                    "odom_delta_m": odom_delta,
                    "odom_cumulative_m": odom_cumulative,
                    "image_path": image_path,
                })
                ground_truth.append({
                    "image_id": image_id,
                    "true_panel_row": panel_row,
                    "true_panel_id": panel_id,
                    "true_east_m": true_east,
                    "true_north_m": true_north,
                    "true_persistent_condition": persistent_condition,
                    "true_transient_overlay": transient_overlay,
                    "true_is_corrupted": is_corrupted,
                    "true_condition_combined": "+".join(issues) if issues else "clean",
                    "true_detected_issues": ",".join(issues),
                })
                capture_counter += 1

            current_time += timedelta(seconds=config.ROW_TRANSIT_S)

    captures_df = pd.DataFrame(captures)
    ground_truth_df = pd.DataFrame(ground_truth)
    route_plan_df = pd.DataFrame(route_plan)
    captures_df = _inject_ingestion_edge_cases(captures_df, ground_truth_df, rng)

    captures_df.to_csv(config.CAPTURES_RAW_PATH, index=False)
    ground_truth_df.to_csv(config.SIM_GROUND_TRUTH_PATH, index=False)
    route_plan_df.to_csv(os.path.join(config.DATA_DIR, "route_plan.csv"), index=False)
    return captures_df, ground_truth_df, route_plan_df


if __name__ == "__main__":
    captures_df, ground_truth_df, route_plan_df = simulate_all()
    print(f"Generated {len(captures_df)} captures across {len(config.MISSION_IDS)} missions "
          f"over {config.NUM_ROWS} rows x {config.PANELS_PER_ROW} panels.")
    print(f"-> {config.FARM_TRUTH_PATH}")
    print(f"-> {config.CAPTURES_RAW_PATH}")
    print(f"-> {config.DATA_DIR}/route_plan.csv ({len(route_plan_df)} route passes)")
    print(f"-> {config.SIM_GROUND_TRUTH_PATH}")
    print(f"-> {config.DATA_DIR}/injected_edge_cases.csv (5 deliberate ingestion edge cases)")
    print(f"-> {config.RAW_IMAGES_DIR}/ ({len(captures_df)} images, minus 1 deliberately missing)")

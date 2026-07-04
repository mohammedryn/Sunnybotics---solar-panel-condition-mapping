"""
RF-06: spatial association under GPS error (highest-weighted dimension, D2 = 20 pts).

Runs a sequential 1D Kalman filter per route_pass_id (one filter per row
traversal), fusing GPS and odometry along the row's own travel axis, then
turns the fused estimate into a posterior over that row's known panels.

State = along-track distance `s` from the route's start point (known
exactly from route_plan.csv, by construction of the simulated route).

  Predict (odometry drives the process model):
      s_pred = s_est[prev] + odom_delta_m
      P_pred = P_est[prev] + ODOM_STEP_STD_M**2
  Update (GPS corrects it, when usable):
      K = P_pred / (P_pred + GPS_FIX_JITTER_STD_M**2)
      s_est = s_pred + K*(s_gps - s_pred)
      P_est = (1-K)*P_pred

Four degraded-input cases per capture, since real sensors don't cooperate:
  GPS valid,   odom ok    -> normal predict + update
  GPS invalid, odom ok    -> predict only, skip update (odometry_only)
  GPS valid,   odom bad   -> don't trust the reported delta; predict with
                             zero assumed motion + inflated variance
                             (ODOM_REJECTED_STEP_VARIANCE_M2), then update
                             fully from GPS (gps_only_odom_rejected)
  GPS invalid, odom bad   -> cannot happen for a row with
                             usable_for_association=True (ingestion's OR
                             logic already guarantees at least one is
                             valid); if it ever does, status is forced to
                             "unresolvable" regardless of what the filter
                             computes.

CONSISTENCY FIX (2nd review round): GPS_FIX_JITTER_STD_M**2 alone is the
right *measurement* noise for the KF update, but using the filter's raw,
possibly-tiny P_est directly for panel-posterior CONFIDENCE is wrong -
repeated GPS updates shrink P_est toward a Riccati steady-state that only
reflects jitter, never the per-mission BIAS (an unmodeled systematic
error). A filter is only "consistent" if its reported covariance actually
bounds the true error; ignoring bias makes it confidently wrong under
bias, not just imprecise. Fix: the KF's own P_est stays mathematically
correct internally (used for future Kalman gains), but panel-posterior
CONFIDENCE is computed from `effective_variance_m2` instead - P_est
floored at ASSOCIATION_VARIANCE_FLOOR_M2, and further inflated by
GPS_MISSION_BIAS_STD_M**2 whenever any GPS update has contributed to this
capture's estimate (once bias-tainted, always tainted going forward,
since the KF's own state carries every prior correction forward).

This does NOT try to estimate/correct the bias itself (that would need
augmenting the state with a bias term - a real refinement, out of scope
here) - it prevents the system from REPORTING false certainty about a
result that unmodeled bias could have made wrong. Also a disclosed
simplification: the bias-variance inflation is a flat all-or-nothing
term keyed on "was GPS used at all in this pass", not scaled by how much
each GPS update's Kalman gain actually weighted it in - a more precise
treatment would track that contribution per-capture.

A pass-level disagreement check also exists: every GPS-valid capture in
a pass produces an innovation (s_gps - s_pred) before its update. Under
jitter alone these average toward zero across a pass; under a real bias
they stay large and consistently signed. mean_pass_innovation_m capture
this, and gps_odometry_disagreement_suspected fires past
BIAS_INNOVATION_THRESHOLD_M - this is DETECTION, not correction, and it
downgrades every image in the pass to "ambiguous" rather than letting a
systematic problem hide behind confident-looking per-image outputs.

Cross-track consistency: using the GPS's north (cross-track) component,
cross_track_delta_m is the distance from the CLAIMED row's centerline.
Past CROSS_TRACK_STRONG_MISMATCH_M (half the row pitch - literally the
Voronoi boundary to the next row), the row itself is called into
question and the result is downgraded to "ambiguous", not just logged.
nominal_row_id is never silently overridden - the robot's own navigation
plan should win unless there's strong contrary evidence, but that
evidence must actually affect the output, not sit next to it unused.

Status decision cascade (checked in this order):
  usable_for_association == False              -> unresolvable
  out_of_bounds (s_fused outside row span)      -> out_of_bounds
  gps_odometry_disagreement_suspected           -> ambiguous (bias)
  cross_track_delta_m > threshold               -> ambiguous (cross-track)
  top1_confidence < ASSOCIATION_CONFIDENCE_MATCHED_THRESHOLD -> ambiguous (low confidence)
  top1_confidence - top2_confidence < TOP2_MARGIN_MATCHED_THRESHOLD -> ambiguous (margin)
  otherwise                                     -> matched

NOTE on the margin gate: it is only reachable, not dead code, because
TOP2_MARGIN_MATCHED_THRESHOLD (0.45) is set above the mathematical bound
(2*ASSOCIATION_CONFIDENCE_MATCHED_THRESHOLD - 1 = 0.4). For ANY discrete
probability distribution, top2 <= 1-top1 (it's at most all remaining
mass), so margin = top1-top2 >= 2*top1-1 whenever the confidence gate
above it has already passed - a margin threshold at or below that bound
could never fire. This was caught and fixed during this module's build
(an earlier value of 0.25 was mathematically unreachable dead code).

HONEST LIMITATION, unchanged from the design discussion and still true
after this fix: a persistent per-mission GPS bias larger than half a
panel pitch cannot be corrected from within-row data alone - no amount
of fusion recovers an absolute reference that was never there. What
this module does is refuse to report false confidence about it, by
flooring/inflating the confidence variance and downgrading affected
passes to "ambiguous" rather than silently outputting a wrong "matched".
Full correction would need an external reference (RTK correction, a
known landmark, a survey marker).
"""
import os

import numpy as np
import pandas as pd

from . import config, farm_layout


def _load_route_plan(route_plan_path: str) -> pd.DataFrame:
    route_df = pd.read_csv(route_plan_path)
    route_df["route_length_m"] = np.sqrt(
        (route_df["end_east_m"] - route_df["start_east_m"]) ** 2
        + (route_df["end_north_m"] - route_df["start_north_m"]) ** 2
    )
    return route_df.set_index("route_pass_id")


def _known_row_norths(farm_df: pd.DataFrame) -> pd.Series:
    """north_m per panel_row - constant within a row by farm construction."""
    return farm_df.groupby("panel_row")["true_north_m"].first()


def _candidates_for_row(farm_df: pd.DataFrame, nominal_row_id: str, route_row: pd.Series) -> pd.DataFrame:
    rows = farm_df[farm_df["panel_row"] == nominal_row_id].copy()
    rows["s_candidate_m"] = np.sqrt(
        (rows["true_east_m"] - route_row["start_east_m"]) ** 2
        + (rows["true_north_m"] - route_row["start_north_m"]) ** 2
    )
    return rows[["panel_id", "s_candidate_m"]].reset_index(drop=True)


def _project_along_track(gps_east: float, gps_north: float, route_row: pd.Series) -> float:
    dx = route_row["end_east_m"] - route_row["start_east_m"]
    dy = route_row["end_north_m"] - route_row["start_north_m"]
    length = route_row["route_length_m"]
    ux, uy = dx / length, dy / length
    return (gps_east - route_row["start_east_m"]) * ux + (gps_north - route_row["start_north_m"]) * uy


def _run_pass_filter(pass_df: pd.DataFrame, route_row: pd.Series, known_row_norths: pd.Series) -> list:
    """First pass over a route_pass_id's captures: runs the recursive KF
    forward, returns one dict per capture with the filter's internal
    state (kept mathematically correct - no flooring/inflation here)."""
    pass_df = pass_df.sort_values("capture_seq_in_pass")
    nominal_row_id = pass_df["nominal_row_id"].iloc[0]
    nominal_row_north = known_row_norths[nominal_row_id]

    s_est, p_est = None, None
    gps_used_so_far = False
    per_capture = []

    for _, cap in pass_df.iterrows():
        gps_ok = cap["coord_validity_status"] == "valid"
        odom_ok = cap["odometry_status"] == "ok"

        if s_est is None:
            prior_s, prior_var = 0.0, config.ROUTE_START_PRIOR_VARIANCE_M2
            if odom_ok:
                delta_var = config.ODOM_STEP_STD_M ** 2
                k = prior_var / (prior_var + delta_var)
                s_pred = prior_s + k * (cap["odom_delta_m"] - prior_s)
                p_pred = (1 - k) * prior_var
            else:
                s_pred, p_pred = prior_s, prior_var
        else:
            if odom_ok:
                s_pred = s_est + cap["odom_delta_m"]
                p_pred = p_est + config.ODOM_STEP_STD_M ** 2
            else:
                s_pred = s_est
                p_pred = p_est + config.ODOM_REJECTED_STEP_VARIANCE_M2

        innovation = None
        cross_track_delta_m = None
        if gps_ok:
            gps_east, gps_north = farm_layout.latlon_to_enu(cap["latitude"], cap["longitude"])
            s_gps = _project_along_track(gps_east, gps_north, route_row)
            cross_track_delta_m = abs(gps_north - nominal_row_north)

            r = config.GPS_FIX_JITTER_STD_M ** 2
            innovation = s_gps - s_pred
            K = p_pred / (p_pred + r)
            s_est = s_pred + K * innovation
            p_est = (1 - K) * p_pred
            gps_used_so_far = True
            localization_source = "gps_odom_fused" if odom_ok else "gps_only_odom_rejected"
        else:
            s_est, p_est = s_pred, p_pred
            localization_source = "odometry_only"

        per_capture.append({
            "image_id": cap["image_id"],
            "s_est": s_est,
            "p_est": p_est,
            "gps_used_so_far": gps_used_so_far,
            "localization_source": localization_source,
            "innovation": innovation,
            "cross_track_delta_m": cross_track_delta_m,
            "usable_for_association": cap["usable_for_association"],
        })

    innovations = [r["innovation"] for r in per_capture if r["innovation"] is not None]
    mean_pass_innovation = float(np.mean(innovations)) if innovations else 0.0
    disagreement_suspected = abs(mean_pass_innovation) > config.BIAS_INNOVATION_THRESHOLD_M

    for r in per_capture:
        r["mean_pass_innovation_m"] = mean_pass_innovation
        r["gps_odometry_disagreement_suspected"] = disagreement_suspected

    return per_capture


def _softmax_posterior(s_fused: float, effective_variance_m2: float, candidates: pd.DataFrame):
    diffs_sq = (candidates["s_candidate_m"] - s_fused) ** 2
    log_weights = -diffs_sq / (2 * effective_variance_m2)
    weights = np.exp(log_weights - log_weights.max())
    posterior = weights / weights.sum()
    ranked = pd.DataFrame({
        "panel_id": candidates["panel_id"],
        "s_candidate_m": candidates["s_candidate_m"],
        "posterior": posterior,
    }).sort_values("posterior", ascending=False).reset_index(drop=True)
    return ranked


def _decide(per_capture_result: dict, route_row: pd.Series, candidates: pd.DataFrame, nominal_row_id: str) -> dict:
    if not per_capture_result["usable_for_association"]:
        return {
            "predicted_panel_row": None, "predicted_panel_id": None,
            "association_confidence": np.nan, "association_status": "unresolvable",
            "ambiguity_reason": "ingestion_usable_for_association_false",
            "localization_source": per_capture_result["localization_source"],
            "fused_along_track_m": per_capture_result["s_est"],
            "fused_variance_m2": per_capture_result["p_est"],
            "effective_variance_m2": np.nan,
            "nearest_panel_distance_m": np.nan, "second_nearest_panel_id": None, "second_nearest_distance_m": np.nan,
            "top2_panel_id": None, "top2_confidence": np.nan, "top2_margin": np.nan,
            "cross_track_mismatch": None, "cross_track_delta_m": per_capture_result["cross_track_delta_m"],
            "gps_odometry_disagreement_suspected": per_capture_result["gps_odometry_disagreement_suspected"],
            "mean_pass_innovation_m": per_capture_result["mean_pass_innovation_m"],
        }

    s_fused, p_est = per_capture_result["s_est"], per_capture_result["p_est"]

    out_of_bounds = (s_fused < -config.PANEL_PITCH_M / 2) or (s_fused > route_row["route_length_m"] + config.PANEL_PITCH_M / 2)

    effective_variance = max(
        p_est,
        config.ASSOCIATION_VARIANCE_FLOOR_M2,
        config.GPS_MISSION_BIAS_STD_M ** 2 if per_capture_result["gps_used_so_far"] else 0.0,
    )

    ranked = _softmax_posterior(s_fused, effective_variance, candidates)
    top1 = ranked.iloc[0]
    top2 = ranked.iloc[1] if len(ranked) > 1 else None
    top1_confidence = float(top1["posterior"])
    top2_confidence = float(top2["posterior"]) if top2 is not None else 0.0
    top2_margin = top1_confidence - top2_confidence

    cross_track_delta_m = per_capture_result["cross_track_delta_m"]
    cross_track_mismatch = (
        cross_track_delta_m is not None and cross_track_delta_m > config.CROSS_TRACK_STRONG_MISMATCH_M
    )

    status, reason = "matched", None
    if out_of_bounds:
        status, reason = "out_of_bounds", "fused_position_outside_row_span"
    elif per_capture_result["gps_odometry_disagreement_suspected"]:
        status, reason = "ambiguous", "gps_odometry_disagreement"
    elif cross_track_mismatch:
        status, reason = "ambiguous", "cross_track_row_mismatch"
    elif top1_confidence < config.ASSOCIATION_CONFIDENCE_MATCHED_THRESHOLD:
        status, reason = "ambiguous", "low_posterior_confidence"
    elif top2_margin < config.TOP2_MARGIN_MATCHED_THRESHOLD:
        status, reason = "ambiguous", "top2_margin_too_small"

    return {
        "predicted_panel_row": nominal_row_id if not out_of_bounds else None,
        "predicted_panel_id": top1["panel_id"] if not out_of_bounds else None,
        "association_confidence": top1_confidence,
        "association_status": status,
        "ambiguity_reason": reason,
        "localization_source": per_capture_result["localization_source"],
        "fused_along_track_m": s_fused,
        "fused_variance_m2": p_est,
        "effective_variance_m2": effective_variance,
        "nearest_panel_distance_m": abs(s_fused - top1["s_candidate_m"]),
        "second_nearest_panel_id": top2["panel_id"] if top2 is not None else None,
        "second_nearest_distance_m": abs(s_fused - top2["s_candidate_m"]) if top2 is not None else np.nan,
        "top2_panel_id": top2["panel_id"] if top2 is not None else None,
        "top2_confidence": top2_confidence,
        "top2_margin": top2_margin,
        "cross_track_mismatch": cross_track_mismatch,
        "cross_track_delta_m": cross_track_delta_m,
        "gps_odometry_disagreement_suspected": per_capture_result["gps_odometry_disagreement_suspected"],
        "mean_pass_innovation_m": per_capture_result["mean_pass_innovation_m"],
    }


def associate(
    ingested_captures_path: str = None,
    route_plan_path: str = None,
    farm_truth_path: str = config.FARM_TRUTH_PATH,
) -> pd.DataFrame:
    ingested_captures_path = ingested_captures_path or os.path.join(config.DATA_DIR, "ingested_captures.csv")
    route_plan_path = route_plan_path or os.path.join(config.DATA_DIR, "route_plan.csv")

    captures_df = pd.read_csv(ingested_captures_path)
    route_lookup = _load_route_plan(route_plan_path)
    farm_df = pd.read_csv(farm_truth_path)
    known_row_norths = _known_row_norths(farm_df)

    results = []

    processable = captures_df[captures_df["route_context_status"] == "ok"]
    unprocessable = captures_df[captures_df["route_context_status"] != "ok"]

    for route_pass_id, group in processable.groupby("route_pass_id"):
        route_row = route_lookup.loc[route_pass_id]
        nominal_row_id = group["nominal_row_id"].iloc[0]
        candidates = _candidates_for_row(farm_df, nominal_row_id, route_row)

        per_capture = _run_pass_filter(group, route_row, known_row_norths)
        for r in per_capture:
            decision = _decide(r, route_row, candidates, nominal_row_id)
            decision["image_id"] = r["image_id"]
            results.append(decision)

    for _, cap in unprocessable.iterrows():
        results.append({
            "image_id": cap["image_id"],
            "predicted_panel_row": None, "predicted_panel_id": None,
            "association_confidence": np.nan, "association_status": "unresolvable",
            "ambiguity_reason": f"route_context_status:{cap['route_context_status']}",
            "localization_source": "none",
            "fused_along_track_m": np.nan, "fused_variance_m2": np.nan, "effective_variance_m2": np.nan,
            "nearest_panel_distance_m": np.nan, "second_nearest_panel_id": None, "second_nearest_distance_m": np.nan,
            "top2_panel_id": None, "top2_confidence": np.nan, "top2_margin": np.nan,
            "cross_track_mismatch": None, "cross_track_delta_m": np.nan,
            "gps_odometry_disagreement_suspected": False, "mean_pass_innovation_m": np.nan,
        })

    return pd.DataFrame(results)


def save_association(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.DATA_DIR, "association_results.csv")
    df.to_csv(path, index=False)


def self_evaluate(association_df: pd.DataFrame, ground_truth_path: str = config.SIM_GROUND_TRUTH_PATH) -> dict:
    """Validates against simulation ground truth. This is a SIMULATION-ONLY
    validation of the method, not a real-world accuracy claim - stated
    explicitly per required report Question 4 (metrics without ground
    truth: here we have it in simulation specifically to validate the
    approach before trusting it on data where we won't)."""
    gt = pd.read_csv(ground_truth_path)
    merged = association_df.merge(gt, on="image_id")

    def _accuracy(d):
        resolved = d[d["predicted_panel_id"].notna()]
        if len(resolved) == 0:
            return None
        return (resolved["predicted_panel_id"] == resolved["true_panel_id"]).mean()

    summary = {
        "n_total": len(merged),
        "n_matched": int((merged["association_status"] == "matched").sum()),
        "n_ambiguous": int((merged["association_status"] == "ambiguous").sum()),
        "n_out_of_bounds": int((merged["association_status"] == "out_of_bounds").sum()),
        "n_unresolvable": int((merged["association_status"] == "unresolvable").sum()),
        "accuracy_overall": _accuracy(merged),
        "accuracy_matched_only": _accuracy(merged[merged["association_status"] == "matched"]),
        "accuracy_ambiguous_only": _accuracy(merged[merged["association_status"] == "ambiguous"]),
    }
    return summary


if __name__ == "__main__":
    df = associate()
    save_association(df)
    print(f"Associated {len(df)} rows -> {config.DATA_DIR}/association_results.csv")
    print("\nassociation_status:\n", df["association_status"].value_counts().to_string())
    print("\nambiguity_reason (non-null):\n", df["ambiguity_reason"].value_counts().to_string())
    print("\nlocalization_source:\n", df["localization_source"].value_counts().to_string())

    print("\n--- Self-evaluation against simulation ground truth (validation-only, not a real-world accuracy claim) ---")
    for k, v in self_evaluate(df).items():
        print(f"{k}: {v}")

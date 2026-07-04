"""
RF-03: condition analysis - orchestration and decision logic ONLY.

This module classifies exclusively from the numeric feature dict that
feature_extraction.py persists to outputs/evidence/<image_id>_features.json
- it never touches raw pixels itself. That separation is deliberate: the
decision function (`_decide_condition`) is a pure function of a features
dict, independently inspectable and testable, and it is architecturally
impossible for the classifier to "cheat" by looking at anything the
feature layer didn't explicitly measure and expose.

Per-issue confidence is a simple, defensible ratio transform, not a
learned score: confidence = clip(raw_score / (2*threshold), 0, 1). At
the calibrated detection threshold, confidence=0.5 ("just barely
detected"); at 2x threshold, confidence=1.0. This is monotonic,
interpretable, and every number in it traces back to a real measurement
in feature_extraction.py.

Decision cascade (revised per review - "uncertain" is a genuine decision
about weak/conflicting evidence, not a default for anything not
maximally confident):
  visual_analysis_status != "ok"        -> condition="uncertain", confidence=NaN
  no issue score passes its threshold   -> condition="clean"
  exactly one issue passes threshold    -> condition=that issue
  >=2 issues pass, top1/top2 margin ok  -> condition=top1 issue (detected_issues
                                            still lists ALL issues that passed,
                                            not just the dominant one)
  >=2 issues pass, margin too small     -> condition="uncertain" (genuine
                                            conflict - detected_issues still
                                            lists both/all likely issues)

Temporal cross-check (optional, confidence-adjusting only - never a
prerequisite for classification): when the SAME predicted_panel_id has a
capture in both missions AND both of those captures independently have
association_status == "matched" (not "ambiguous" - a stricter bar than
originally proposed, specifically because a wrongly-grouped ambiguous
pair could reinforce an association error into a false confidence
adjustment), a shadow-like signal present in only one mission's image is
evidence it's transient; a dirt-like signal present in both is evidence
it's persistent. This adjusts condition_confidence; it never blocks or
overrides the independent single-image classification, preserving the
principle that spatial confidence and visual diagnosis are separate axes.
"""
import json
import os

import numpy as np
import pandas as pd

from . import config, feature_extraction as fe


ISSUE_FEATURE_MAP = {
    "dirt": "dirt_area_ratio",
    "shadow": "shadow_area_ratio",
    "glare": "glare_area_ratio",
    "damage": "damage_line_density",
}


def _issue_confidence(raw_score: float, threshold: float) -> float:
    return float(np.clip(raw_score / (2 * threshold), 0.0, 1.0))


def _decide_condition(features: dict) -> dict:
    """Pure function: features dict -> (condition, condition_confidence,
    detected_issues). No image access, no side effects. Always uses
    config.ISSUE_DETECTION_THRESHOLDS - this is the zero-shot,
    synthetic-calibrated rule classifier, unconditionally, for both
    modes. (An earlier version of this build tried recalibrating these
    thresholds per-batch for external mode; that made real-photo
    predictions non-degenerate but WORSE than naive baselines - see
    src/external_classifier.py for the supervised replacement, which is
    the real external-mode metric now.)"""
    confidences = {
        issue: _issue_confidence(features[feat_name], config.ISSUE_DETECTION_THRESHOLDS[issue])
        for issue, feat_name in ISSUE_FEATURE_MAP.items()
    }
    detected_issues = [
        issue for issue, feat_name in ISSUE_FEATURE_MAP.items()
        if features[feat_name] > config.ISSUE_DETECTION_THRESHOLDS[issue]
    ]

    if not detected_issues:
        return {
            "condition": "clean",
            "condition_confidence": 1.0 - max(confidences.values()),
            "detected_issues": [],
        }

    ranked = sorted(detected_issues, key=lambda i: confidences[i], reverse=True)
    top1 = ranked[0]
    top1_conf = confidences[top1]

    if len(ranked) == 1:
        return {"condition": top1, "condition_confidence": top1_conf, "detected_issues": detected_issues}

    top2_conf = confidences[ranked[1]]
    margin = top1_conf - top2_conf
    if margin < config.CONDITION_TOP2_MARGIN_THRESHOLD:
        return {
            "condition": "uncertain",
            "condition_confidence": top1_conf,
            "detected_issues": detected_issues,
        }
    return {"condition": top1, "condition_confidence": top1_conf, "detected_issues": detected_issues}


def _visual_analysis_status(row: pd.Series) -> tuple:
    if row["image_load_status"] == "missing_file":
        return "image_missing", "missing_file"
    if row["image_load_status"] == "corrupted_unreadable":
        return "image_corrupt", "corrupted_unreadable"
    if row["image_content_status"] != "ok":
        return "unusable", row["image_content_status"]
    return "ok", None


def _temporal_adjust(image_id: str, row_features: dict, condition: str, confidence: float,
                      association_df: pd.DataFrame, captures_df: pd.DataFrame,
                      features_by_image: dict) -> float:
    if condition not in ("dirt", "shadow"):
        return confidence

    this_assoc = association_df[association_df["image_id"] == image_id]
    if this_assoc.empty or this_assoc.iloc[0]["association_status"] != config.TEMPORAL_CROSSCHECK_REQUIRED_STATUS:
        return confidence
    predicted_panel_id = this_assoc.iloc[0]["predicted_panel_id"]
    this_mission = captures_df.loc[captures_df["image_id"] == image_id, "mission_id"].iloc[0]

    partner_assoc = association_df[
        (association_df["predicted_panel_id"] == predicted_panel_id)
        & (association_df["association_status"] == config.TEMPORAL_CROSSCHECK_REQUIRED_STATUS)
        & (association_df["image_id"] != image_id)
    ]
    if partner_assoc.empty:
        return confidence
    partner_id = partner_assoc.iloc[0]["image_id"]
    partner_mission = captures_df.loc[captures_df["image_id"] == partner_id, "mission_id"]
    if partner_mission.empty or partner_mission.iloc[0] == this_mission:
        return confidence  # only cross-check across DIFFERENT missions
    partner_features = features_by_image.get(partner_id)
    if partner_features is None:
        return confidence

    partner_has_same_signal = partner_features[ISSUE_FEATURE_MAP[condition]] > config.ISSUE_DETECTION_THRESHOLDS[condition]

    if condition == "shadow":
        # transient signal absent in the other mission -> more likely real shadow -> boost
        return min(1.0, confidence * 1.2) if not partner_has_same_signal else confidence * 0.9
    else:  # dirt
        # persistent signal present in both missions -> more likely real dirt -> boost
        return min(1.0, confidence * 1.2) if partner_has_same_signal else confidence * 0.9


def analyze(
    ingested_captures_path: str = None,
    association_results_path: str = None,
) -> pd.DataFrame:
    ingested_captures_path = ingested_captures_path or os.path.join(config.DATA_DIR, "ingested_captures.csv")
    association_results_path = association_results_path or os.path.join(config.DATA_DIR, "association_results.csv")

    captures_df = pd.read_csv(ingested_captures_path)
    association_df = pd.read_csv(association_results_path)

    results = []
    features_by_image = {}

    for _, row in captures_df.iterrows():
        status, reason = _visual_analysis_status(row)
        if status != "ok":
            results.append({
                "image_id": row["image_id"], "condition": "uncertain", "condition_confidence": np.nan,
                "detected_issues": [], "visual_analysis_status": status, "unusable_reason": reason,
                "evidence_bbox_or_mask_path": None, "feature_summary_json": None,
            })
            continue

        try:
            features, mask_paths = fe.extract_features(row["image_path"], row["image_id"], save_evidence=True)
        except Exception as e:
            results.append({
                "image_id": row["image_id"], "condition": "uncertain", "condition_confidence": np.nan,
                "detected_issues": [], "visual_analysis_status": "unusable", "unusable_reason": f"error:{e}",
                "evidence_bbox_or_mask_path": None, "feature_summary_json": None,
            })
            continue

        if features["laplacian_variance"] < config.BLUR_LAPLACIAN_VARIANCE_THRESHOLD:
            results.append({
                "image_id": row["image_id"], "condition": "uncertain", "condition_confidence": np.nan,
                "detected_issues": [], "visual_analysis_status": "unusable", "unusable_reason": "blurry",
                "evidence_bbox_or_mask_path": None, "feature_summary_json": mask_paths.get("features_path"),
            })
            continue

        features_by_image[row["image_id"]] = features
        decision = _decide_condition(features)
        evidence_path = mask_paths.get(f"{decision['condition']}_mask_path") if decision["condition"] in ISSUE_FEATURE_MAP else None

        results.append({
            "image_id": row["image_id"],
            "condition": decision["condition"],
            "condition_confidence": decision["condition_confidence"],
            "detected_issues": decision["detected_issues"],
            "visual_analysis_status": "ok",
            "unusable_reason": None,
            "evidence_bbox_or_mask_path": evidence_path,
            "feature_summary_json": mask_paths.get("features_path"),
            "_features": features,
        })

    for r in results:
        if r.get("condition") in ("dirt", "shadow") and "_features" in r:
            r["condition_confidence"] = _temporal_adjust(
                r["image_id"], r["_features"], r["condition"], r["condition_confidence"],
                association_df, captures_df, features_by_image,
            )
        r.pop("_features", None)

    df = pd.DataFrame(results)
    df["detected_issues"] = df["detected_issues"].apply(lambda v: ",".join(v) if isinstance(v, list) else v)
    return df


def save_results(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.DATA_DIR, "condition_results.csv")
    df.to_csv(path, index=False)


def self_evaluate(condition_df: pd.DataFrame, ground_truth_path: str = None) -> dict:
    """Simulation-only validation against ground truth - same discipline
    as associate_panels.py's self_evaluate. Not a real-world accuracy claim.
    Synthetic mode only; external mode's evaluation lives in
    external_dataset.py since its ground truth (clean/damaged folder
    labels) has a different shape than simulated per-panel truth."""
    ground_truth_path = ground_truth_path or config.SIM_GROUND_TRUTH_PATH
    gt = pd.read_csv(ground_truth_path)
    merged = condition_df.merge(gt, on="image_id")

    # simulate_mission.py's true_persistent_condition uses "dirty"/"damaged"
    # (matching config.PERSISTENT_CONDITION_WEIGHTS keys); the classifier's
    # own vocabulary (and image_synth.py's issue tags) use "dirt"/"damage".
    # This mapping is the bridge - NOT a re-derivation of ground truth.
    PERSISTENT_TO_CLASSIFIER_LABEL = {"dirty": "dirt", "damaged": "damage", "clean": "clean"}

    def _true_label(row):
        if row["true_is_corrupted"]:
            return "uncertain"
        if row["true_persistent_condition"] != "clean":
            return PERSISTENT_TO_CLASSIFIER_LABEL[row["true_persistent_condition"]]
        if row["true_transient_overlay"] != "none":
            return row["true_transient_overlay"]  # already "shadow"/"glare"
        return "clean"

    merged["true_label"] = merged.apply(_true_label, axis=1)
    resolved = merged[merged["visual_analysis_status"] == "ok"]
    accuracy = (resolved["condition"] == resolved["true_label"]).mean() if len(resolved) else None

    per_category = {}
    for cond in ["clean", "dirt", "shadow", "glare", "damage"]:
        sub = resolved[resolved["true_label"] == cond]
        if len(sub) > 0:
            per_category[cond] = float((sub["condition"] == cond).mean())

    return {
        "n_total": len(merged),
        "n_ok": len(resolved),
        "n_uncertain_or_unusable": int((merged["condition"] == "uncertain").sum()),
        "accuracy_overall": accuracy,
        "accuracy_per_true_category": per_category,
    }


if __name__ == "__main__":
    df = analyze()
    save_results(df)
    print(f"Analyzed {len(df)} rows -> {config.DATA_DIR}/condition_results.csv")
    print("\ncondition:\n", df["condition"].value_counts().to_string())
    print("\nvisual_analysis_status:\n", df["visual_analysis_status"].value_counts().to_string())

    print("\n--- Self-evaluation against simulation ground truth (validation-only) ---")
    for k, v in self_evaluate(df).items():
        print(f"{k}: {v}")

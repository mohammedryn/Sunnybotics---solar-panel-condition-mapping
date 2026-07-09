"""
RF-04: priority scoring, tied to operational O&M decisions - NOT a
generic confidence-weighted class label.

Branches on `detected_issues` (the full multi-label list from
condition_analysis.py), not on `condition` (the single dominant label
picked by confidence). This matters: condition_analysis.py's classifier
picks `condition` by confidence, so a panel with both a weaker damage
signal and a stronger dirt signal can get `condition="dirt"` reported as
dominant even though damage is present in `detected_issues`.
Operationally that damage signal must still win - a moderately-confident
crack indicator cannot be buried under a more-confident dirt reading.

A real bug lived here until caught: the damage branch scaled its score
using `condition_confidence` - the DOMINANT condition's confidence, not
damage's own. For a row where glare/shadow was clearly dominant
(confidence ~1.0) and damage only weakly cleared its own threshold as a
secondary signal, that meant the damage score - and the wording, which
said "Damage detected (confidence=1.00)" - both overstated it, producing
a `glare` image landing at critical/inspect with text that read as if
damage were confidently confirmed. `_damage_confidence()` computes
damage's own confidence from its own feature regardless of dominance,
and the reason text distinguishes "damage is the primary finding" from
"damage is a secondary, unconfirmed signal worth a look" - the asymmetric
risk argument doesn't require pretending the system is more certain than
it is.

Branch order (operational risk, checked in this order - first match wins):
  visual_analysis_status != "ok"   -> recapture (can't assess at all;
                                       an operations problem, not a
                                       condition problem)
  "damage" in detected_issues      -> inspect, score floors HIGH
                                       regardless of confidence. This is
                                       the ONE branch that overrides the
                                       classifier's own dominant
                                       `condition` label - deliberately,
                                       since a moderately-confident
                                       damage signal must not be buried
                                       under a more-confident dirt/shadow
                                       reading.
  condition == "uncertain"         -> human_review, moderate score,
                                       never "clean". Checked BEFORE the
                                       dirt/shadow branches on purpose:
                                       verified against this project's
                                       own data that every genuinely
                                       ambiguous case (top1/top2 conflict)
                                       has "dirt" present in
                                       detected_issues (16/20 cases) - if
                                       dirt were checked first, the
                                       "uncertain" branch would never
                                       fire at all, silently contradicting
                                       "not automatic cleaning". Damage
                                       is still checked first above,
                                       since that override is explicit
                                       and intentional; dirt/shadow are
                                       not given that same override.
  condition == "dirt"              -> clean, score scales with MEASURED
                                       AREA (not confidence)
  condition in ("shadow","glare")  -> human_review, flat low-medium
                                       score (probably transient, but
                                       panel underneath wasn't assessed)
  otherwise (clean)                -> none, low score

Priority score is deliberately independent of association_status (RF-06)
- spatial confidence and visual diagnosis are separate axes. But a
high-priority finding on an `ambiguous`/`unresolvable` association can't
actually be dispatched to a specific panel_id yet; that's surfaced as a
note in priority_reason, not folded into the score itself.
"""
import json
import os

import numpy as np
import pandas as pd

from . import condition_analysis, config


def _priority_band(score: float) -> str:
    if score <= config.PRIORITY_BAND_BOUNDS["low"]:
        return "low"
    if score <= config.PRIORITY_BAND_BOUNDS["medium"]:
        return "medium"
    if score <= config.PRIORITY_BAND_BOUNDS["high"]:
        return "high"
    return "critical"


def _damage_confidence(features: dict) -> float:
    """Damage's OWN confidence, computed the same way condition_analysis.py
    computes it - regardless of whether damage ended up as the dominant
    `condition`. Using the dominant condition's confidence here was a real
    bug: a row with a clearly-dominant glare/shadow reading (confidence
    ~1.0) and only a WEAK secondary damage signal would score as if damage
    itself were maximally confident, because the two were conflated."""
    raw = features.get(condition_analysis.ISSUE_FEATURE_MAP["damage"], 0.0)
    return condition_analysis._issue_confidence(raw, config.ISSUE_DETECTION_THRESHOLDS["damage"])


def _load_features(feature_summary_json_path) -> dict:
    if not isinstance(feature_summary_json_path, str) or not os.path.exists(feature_summary_json_path):
        return {}
    with open(feature_summary_json_path) as f:
        return json.load(f)


def _association_caveat(image_id: str, association_df: pd.DataFrame) -> str:
    if association_df is None:
        return ""
    match = association_df[association_df["image_id"] == image_id]
    if match.empty:
        return ""
    status = match.iloc[0]["association_status"]
    if status in ("ambiguous", "unresolvable", "out_of_bounds"):
        return f" Note: panel association is '{status}' - verify panel identity before dispatch."
    return ""


def _score_row(row: pd.Series, association_df: pd.DataFrame) -> dict:
    caveat = _association_caveat(row["image_id"], association_df)

    if row["visual_analysis_status"] != "ok":
        return {
            "cleaning_priority_score": config.UNUSABLE_RECAPTURE_PRIORITY_SCORE,
            "priority_band": _priority_band(config.UNUSABLE_RECAPTURE_PRIORITY_SCORE),
            "priority_reason": f"Image {row['visual_analysis_status']} ({row['unusable_reason']}) - "
                                f"cannot assess condition; recapture required before any other action.",
            "recommended_action": "recapture",
        }

    detected_issues = row["detected_issues"].split(",") if isinstance(row["detected_issues"], str) and row["detected_issues"] else []
    features = _load_features(row["feature_summary_json"])
    confidence = row["condition_confidence"]

    if "damage" in detected_issues:
        damage_confidence = _damage_confidence(features)
        score = min(100.0, config.DAMAGE_PRIORITY_FLOOR + config.DAMAGE_PRIORITY_CONFIDENCE_RANGE * damage_confidence)
        is_dominant = row["condition"] == "damage"

        if is_dominant:
            reason = (
                f"Damage detected (confidence={damage_confidence:.2f}) - inspection prioritized regardless of "
                f"confidence given asymmetric risk of missed structural damage."
                + (" Dirt also present; address during inspection visit." if "dirt" in detected_issues else "")
            )
        else:
            # Damage is not the dominant finding - condition_confidence
            # belongs to the actual dominant condition (e.g. glare/shadow),
            # not to damage. Worded to be explicit that this is a secondary,
            # unconfirmed signal, not "this image is confidently damaged" -
            # otherwise a glare image landing at critical/inspect reads as
            # the system being overconfident rather than appropriately
            # cautious.
            reason = (
                f"Primary visual finding is {row['condition']} (confidence={confidence:.2f}), but a possible "
                f"damage signal was also detected (confidence={damage_confidence:.2f}) - inspection recommended "
                f"due to asymmetric risk, not a confirmed diagnosis of damage."
            )
        reason += caveat

        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": reason,
            "recommended_action": "inspect",
        }

    if row["condition"] == "uncertain":
        score = config.UNCERTAIN_PRIORITY_SCORE
        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": (
                f"Conflicting or weak visual evidence (candidates: {row['detected_issues']}) - condition "
                f"genuinely uncertain; human review recommended, no automatic cleaning dispatch." + caveat
            ),
            "recommended_action": "human_review",
        }

    if row["condition"] == "dirt":
        area = features.get("dirt_area_ratio", 0.0)
        severity = np.clip(area / config.DIRT_SEVERITY_SATURATION_RATIO, 0.0, 1.0)
        score = config.DIRT_PRIORITY_MIN + (config.DIRT_PRIORITY_MAX - config.DIRT_PRIORITY_MIN) * severity
        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": (
                f"Dirt detected covering {area:.1%} of panel area - cleaning priority scaled to soiling "
                f"severity, not classification confidence." + caveat
            ),
            "recommended_action": "clean",
        }

    if row["condition"] in ("shadow", "glare"):
        score = config.SHADOW_GLARE_PRIORITY_SCORE
        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": (
                f"{row['condition'].capitalize()} detected - likely a transient optical effect, not a "
                f"persistent panel condition. Panel underneath could not be fully assessed; recommend "
                f"review or recapture at a different time rather than a cleaning dispatch." + caveat
            ),
            "recommended_action": "human_review",
        }

    score = config.CLEAN_PRIORITY_SCORE
    return {
        "cleaning_priority_score": score,
        "priority_band": _priority_band(score),
        "priority_reason": "No condition issues detected." + caveat,
        "recommended_action": "none",
    }


def _score_external_row(row: pd.Series) -> dict:
    """External-mode priority scoring - deliberately NOT a call into
    _score_row(). That function's branch order is keyed on
    detected_issues, which comes from the same classical CV thresholds
    already shown (in the report's own Section III) not to transfer to
    real photos - confirmed empirically this session that "damage"
    appears in detected_issues for all 129 real rows regardless of true
    label, which is exactly why priority scoring was flat before this
    fix. This function is keyed only on the promoted condition (the
    supervised classifier's call), reusing the same config constants and
    formula shapes as _score_row for philosophical consistency, applied
    to the classifier that's actually reliable on real data instead."""
    if row["visual_analysis_status"] != "ok":
        return {
            "cleaning_priority_score": config.UNUSABLE_RECAPTURE_PRIORITY_SCORE,
            "priority_band": _priority_band(config.UNUSABLE_RECAPTURE_PRIORITY_SCORE),
            "priority_reason": f"Image {row['visual_analysis_status']} ({row['unusable_reason']}) - "
                                f"cannot assess condition; recapture required before any other action.",
            "recommended_action": "recapture",
        }

    confidence = row["condition_confidence"]
    if row["condition"] == "damaged":
        score = min(100.0, config.DAMAGE_PRIORITY_FLOOR + config.DAMAGE_PRIORITY_CONFIDENCE_RANGE * confidence)
        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": f"Supervised classifier called this panel damaged (confidence={confidence:.2f}) "
                                f"- inspection prioritized given asymmetric risk of missed structural damage.",
            "recommended_action": "inspect",
        }

    if row["condition"] == "clean":
        score = config.CLEAN_PRIORITY_SCORE
        return {
            "cleaning_priority_score": score,
            "priority_band": _priority_band(score),
            "priority_reason": f"Supervised classifier called this panel clean (confidence={confidence:.2f}).",
            "recommended_action": "none",
        }

    # Defensive fallback - should not be reached given promote_to_primary_condition's
    # contract (every "ok" row gets a clean/damaged call), but never silently
    # guess at a priority for a condition value this function doesn't recognize.
    score = config.UNCERTAIN_PRIORITY_SCORE
    return {
        "cleaning_priority_score": score,
        "priority_band": _priority_band(score),
        "priority_reason": f"Unrecognized condition '{row['condition']}' for external mode - human review recommended.",
        "recommended_action": "human_review",
    }


def score_priorities(
    condition_results_path: str = None,
    association_results_path: str = None,
    mode: str = "synthetic",
) -> pd.DataFrame:
    condition_results_path = condition_results_path or os.path.join(config.DATA_DIR, "condition_results.csv")
    association_results_path = association_results_path or os.path.join(config.DATA_DIR, "association_results.csv")

    condition_df = pd.read_csv(condition_results_path)
    association_df = pd.read_csv(association_results_path) if os.path.exists(association_results_path) else None

    rows = []
    for _, row in condition_df.iterrows():
        scored = _score_external_row(row) if mode == "external" else _score_row(row, association_df)
        scored["image_id"] = row["image_id"]
        rows.append(scored)

    return pd.DataFrame(rows)[
        ["image_id", "cleaning_priority_score", "priority_band", "priority_reason", "recommended_action"]
    ]


def save_priorities(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.DATA_DIR, "priority_results.csv")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = score_priorities()
    save_priorities(df)
    print(f"Scored {len(df)} rows -> {config.DATA_DIR}/priority_results.csv")
    print("\npriority_band:\n", df["priority_band"].value_counts().to_string())
    print("\nrecommended_action:\n", df["recommended_action"].value_counts().to_string())
    print("\nscore stats:\n", df["cleaning_priority_score"].describe().to_string())

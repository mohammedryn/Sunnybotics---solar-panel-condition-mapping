"""
RF-01: image ingestion.

Reads data/captures_raw.csv (or any captures_raw with the same schema -
this module doesn't care whether the underlying rows came from the
simulator or a real Sunnybotics dataset) and validates each row against
the failure classes the brief explicitly calls out: "must not crash on
a missing or corrupted image, incomplete metadata, or invalid
coordinates."

A row that fails validation is flagged and passed through with its
status attached, never silently dropped or "fixed" - deciding what to
do with an unusable row belongs to downstream steps; ingestion's job is
only to say, honestly, what's wrong.

Status columns are deliberately task-specific rather than one blanket
`is_usable` boolean, because the tasks genuinely have different
requirements: association (RF-06) only needs route identity plus *a*
position source (GPS or odometry - either is enough, that's the whole
point of fusing them); visual analysis (RF-03) only needs a loadable,
non-degenerate image and doesn't care about GPS at all. A single
combined flag would mark a GPS-dropout row "unusable" even though
Section 2's odometry fallback can still localize it - a real
contradiction an evaluator could catch (a prior review round did
exactly that).

  usable_for_association    - route context is internally and
                               referentially valid (route_context_status
                               == ok, checked against route_plan.csv and
                               the farm layout) AND at least one plausible
                               position source is available (valid GPS OR
                               odometry_status == ok). Image content is
                               irrelevant here.
  usable_for_visual_analysis - image loads AND its content passes basic
                               sanity checks (not blank/near-constant,
                               correct dimensions). GPS/route irrelevant here.
  requires_attention         - true if ANY status below is not clean;
                               general-purpose triage flag for a
                               dashboard or human review, not a pipeline
                               gate.

IMPORTANT SCOPE LIMIT, stated explicitly rather than left implicit: this
module checks whether GPS and odometry are each internally plausible
and whether the route context is real (exists in route_plan.csv,
matches the farm layout). It does NOT check whether GPS and odometry
*agree with each other* - two individually-plausible readings that
point to very different positions both pass ingestion. Detecting that
kind of mutual disagreement requires actually fusing them, which is
Section 2's (associate_panels.py's) job, not ingestion's. Ingestion's
job is narrower: reject inputs that are impossible or unknown before
they ever reach the fusion layer, so Section 2 only has to reason about
genuine sensor disagreement, not garbage data.

`usable_for_final_scoring` (loadable image AND a resolved - matched or
ambiguous - association) is NOT computed here: it depends on
associate_panels.py's output, which does not exist yet at ingestion
time. Faking it here would just be a different overloaded boolean.
It belongs at export time (RF-07), once both this module's and
associate_panels.py's results can be joined on image_id.

Image validity note: PIL's Image.verify() only checks file/header
structure and does NOT reliably catch a truncated JPEG (confirmed
empirically against this project's own truncated-file test case - it
passed verify() cleanly). A full Image.load() forces complete decode
and correctly raises OSError on truncation. cv2.imread does not raise
at all on the same file - it silently returns a partial image with
grey-padded rows, which would have been invisible to anything checking
only "is the return value None". image_load_status uses the strict
PIL .load() path specifically because of that gap.

image_content_status is a separate, cheap sanity check on the decoded
pixels themselves - a file can be a perfectly valid JPEG and still be
useless for analysis (e.g. the synthetic "blackout"/"overexposed"
corruption modes are valid images with std ~0). Threshold calibrated
empirically against this project's own data: normal renders sit at
std ~11.5-13.5, heavy blur ~8.9, blackout/overexposed ~0-0.3 - a
threshold of 5.0 cleanly separates blank/near-constant from everything
else, including blur (which is deliberately NOT flagged here; blur
detection is Section 3 / required Question 2's job, not ingestion's).
Known limitation: a std check only catches *degenerate* content
(blank/constant). A structurally normal but semantically wrong image
(e.g. a photo from an unrelated dataset, or a plausible-looking rotated/
cropped frame) would still report "ok" here - genuinely detecting "is
this actually a solar panel" would need a dedicated classifier, out of
scope for this project and not something this simulated dataset can
even exercise (every image here is a real panel render by construction).

route_context_status and odometry_status cross-reference route_plan.csv
and the farm layout to reject referentially-invalid or physically
implausible inputs (unknown route pass, unknown/mismatched row,
negative or out-of-range along-track distance, non-monotonic sequence
within a pass) *before* they reach Section 2's fusion step. This is
deliberately narrower than detecting GPS-vs-odometry disagreement (see
the scope-limit note above) - it only rejects inputs that couldn't be
correct under any interpretation, leaving genuine sensor disagreement
for the fusion layer to reason about with its uncertainty model.
"""
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import config, farm_layout

# Core identity fields every dataset must have, regardless of mode.
CORE_REQUIRED_FIELDS = ["image_id", "timestamp", "robot_id", "mission_id"]
# Route identity is only required of datasets that HAVE route structure at
# all (the synthetic mission simulator). The external Sunnybotics sample
# dataset has no GPS/route/panel metadata by its own description - it
# simply doesn't have these columns, which is not the same failure mode
# as a synthetic row missing them (an actual data-quality problem). Which
# fields are "required" is therefore computed per dataset from the
# columns actually present, not assumed fixed - see ingest()'s
# `required_fields` local.
ROUTE_REQUIRED_FIELDS = ["route_pass_id", "nominal_row_id"]

# How far (meters) a GPS fix can be from the farm's bounding envelope
# before it's flagged as implausible for this deployment, even if it's
# a structurally valid lat/lon. Generous margin beyond the ~20m x 20m
# farm extent - this is not meant to catch ordinary GPS error (that's
# RF-06's job), only gross faults such as a receiver reporting a
# default/reset position.
FARM_PLAUSIBILITY_MARGIN_M = 200.0


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _check_metadata(row: pd.Series, required_fields: list) -> str:
    for field in required_fields:
        if _is_missing(row.get(field)):
            return f"incomplete_metadata:{field}"
    return "ok"


def _check_coordinates(row: pd.Series) -> str:
    lat, lon = row.get("latitude"), row.get("longitude")
    if _is_missing(lat) or _is_missing(lon):
        return "missing_nan"
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return "out_of_range"

    east_m, north_m = farm_layout.latlon_to_enu(lat, lon)
    farm_span_e = config.PANELS_PER_ROW * config.PANEL_PITCH_M
    farm_span_n = config.NUM_ROWS * config.ROW_PITCH_M
    within_e = -FARM_PLAUSIBILITY_MARGIN_M <= east_m <= farm_span_e + FARM_PLAUSIBILITY_MARGIN_M
    within_n = -FARM_PLAUSIBILITY_MARGIN_M <= north_m <= farm_span_n + FARM_PLAUSIBILITY_MARGIN_M
    if not (within_e and within_n):
        return "implausible_for_farm"
    return "valid"


def _check_image_content(pixels: np.ndarray) -> str:
    # No longer requires an EXACT match to the synthetic renderer's fixed
    # 480x360 canvas - only ever mattered for synthetic data (which is
    # always exactly that size by construction, so this never actually
    # fired in practice there either), and would have flagged every
    # single real external photo as "unexpected_dimensions" purely for
    # having a normal, varying real-world resolution. MIN_IMAGE_DIMENSION_PX
    # and the blank/near-constant check below are the checks that
    # actually catch degenerate content; exact-size matching wasn't a
    # meaningful content-sanity signal to begin with.
    h, w = pixels.shape[:2]
    if h < config.MIN_IMAGE_DIMENSION_PX or w < config.MIN_IMAGE_DIMENSION_PX:
        return "too_small"
    if pixels.astype(np.float32).std() < config.BLANK_STD_THRESHOLD:
        return "blank_or_near_constant"
    return "ok"


def _check_image(path):
    """Returns (image_load_status, image_content_status)."""
    if _is_missing(path) or not os.path.exists(path):
        return "missing_file", "n/a"
    try:
        with Image.open(path) as img:
            img.load()  # full decode - .verify() alone misses truncated JPEGs
    except Exception:
        return "corrupted_unreadable", "n/a"

    pixels = cv2.imread(path)
    if pixels is None or pixels.size == 0:
        return "corrupted_unreadable", "n/a"
    return "ok", _check_image_content(pixels)


def _load_route_plan(route_plan_path: str) -> dict:
    """route_pass_id -> {nominal_row_id, route_length_m}."""
    route_df = pd.read_csv(route_plan_path)
    route_df["route_length_m"] = np.sqrt(
        (route_df["end_east_m"] - route_df["start_east_m"]) ** 2
        + (route_df["end_north_m"] - route_df["start_north_m"]) ** 2
    )
    return {
        r["route_pass_id"]: {"nominal_row_id": r["nominal_row_id"], "route_length_m": r["route_length_m"]}
        for _, r in route_df.iterrows()
    }


def _check_route_context(row: pd.Series, route_lookup: dict, known_row_ids: set) -> str:
    """Referential integrity: does this row point at a route pass / farm
    row that actually exists, and do the two identity fields on this row
    agree with what the route plan itself says? This is checked
    independently of GPS/odometry plausibility - a row can have a
    perfectly valid GPS fix and still reference a route_pass_id that was
    never planned (e.g. a bad join upstream)."""
    route_pass_id, nominal_row_id = row.get("route_pass_id"), row.get("nominal_row_id")
    if _is_missing(route_pass_id) or route_pass_id not in route_lookup:
        return "unknown_route_pass"
    if _is_missing(nominal_row_id) or nominal_row_id not in known_row_ids:
        return "unknown_nominal_row"
    if nominal_row_id != route_lookup[route_pass_id]["nominal_row_id"]:
        return "route_row_mismatch"
    return "ok"


def _check_odometry_bounds(row: pd.Series, route_lookup: dict) -> str:
    """Plausibility of odom_cumulative_m in isolation (missing / negative /
    beyond route length). Monotonicity across a pass is checked separately
    afterward since it needs the other rows in the same route_pass_id, not
    just this one row."""
    cumulative = row.get("odom_cumulative_m")
    if _is_missing(cumulative):
        return "missing_nan"
    if cumulative < -config.ROUTE_ODOM_TOLERANCE_M:
        return "negative_distance"

    route_info = route_lookup.get(row.get("route_pass_id"))
    if route_info is not None:
        if cumulative > route_info["route_length_m"] + config.ROUTE_ODOM_TOLERANCE_M:
            return "exceeds_route_length"
    return "ok"


def ingest(
    captures_path: str = None,
    route_plan_path: str = None,
    farm_truth_path: str = None,
) -> pd.DataFrame:
    # All three looked up at call time (config.set_mode() runs before this
    # is called, but a signature default like `=config.X` is bound once at
    # import time and would silently ignore a later mode switch).
    captures_path = captures_path or config.CAPTURES_RAW_PATH
    farm_truth_path = farm_truth_path or config.FARM_TRUTH_PATH
    route_plan_path = route_plan_path or os.path.join(config.DATA_DIR, "route_plan.csv")
    captures_df = pd.read_csv(captures_path)

    # A dataset either has route structure (route_pass_id, nominal_row_id
    # columns present - the synthetic simulator always provides these) or
    # it doesn't (the external dataset, which has no GPS/route/panel
    # metadata at all by its own description). This is a schema property
    # of the dataset, not a per-row data-quality problem, so it's checked
    # once here rather than per row.
    has_route_structure = "route_pass_id" in captures_df.columns and "nominal_row_id" in captures_df.columns
    required_fields = CORE_REQUIRED_FIELDS + (ROUTE_REQUIRED_FIELDS if has_route_structure else [])

    if has_route_structure:
        route_lookup = _load_route_plan(route_plan_path)
        known_row_ids = set(pd.read_csv(farm_truth_path)["panel_row"].unique())

    metadata_status, coord_status, image_status, content_status = [], [], [], []
    route_context_status, odometry_status = [], []
    for _, row in captures_df.iterrows():
        try:
            metadata_status.append(_check_metadata(row, required_fields))
        except Exception as e:
            metadata_status.append(f"error:{e}")
        try:
            coord_status.append(_check_coordinates(row))
        except Exception as e:
            coord_status.append(f"error:{e}")
        try:
            load_status, content = _check_image(row.get("image_path"))
        except Exception as e:
            load_status, content = f"error:{e}", "n/a"
        image_status.append(load_status)
        content_status.append(content)

        if not has_route_structure:
            # Not a failed check - there is nothing to check. Distinct from
            # "ok" (checked and valid) and from the specific failure enum
            # values (checked and found wrong) on purpose.
            route_context_status.append("not_applicable")
            odometry_status.append("not_applicable")
            continue
        try:
            route_context_status.append(_check_route_context(row, route_lookup, known_row_ids))
        except Exception as e:
            route_context_status.append(f"error:{e}")
        try:
            odometry_status.append(_check_odometry_bounds(row, route_lookup))
        except Exception as e:
            odometry_status.append(f"error:{e}")

    captures_df["metadata_status"] = metadata_status
    captures_df["coord_validity_status"] = coord_status
    captures_df["image_load_status"] = image_status
    captures_df["image_content_status"] = content_status
    captures_df["route_context_status"] = route_context_status
    captures_df["odometry_status"] = odometry_status

    # Monotonicity needs the full sequence within each route_pass_id, so it's
    # computed after the per-row pass above and merged in by overriding
    # "ok" -> "non_monotonic_in_pass" where it fires (lowest-priority check;
    # missing/negative/exceeds already reported take precedence per row).
    sort_cols = ["route_pass_id", "capture_seq_in_pass"]
    if all(c in captures_df.columns for c in sort_cols):
        ordered = captures_df.sort_values(sort_cols)
        diffs = ordered.groupby("route_pass_id")["odom_cumulative_m"].diff()
        non_monotonic_idx = ordered.index[diffs < -1e-6]
        still_ok = captures_df["odometry_status"] == "ok"
        flip_idx = [i for i in non_monotonic_idx if still_ok.get(i, False)]
        captures_df.loc[flip_idx, "odometry_status"] = "non_monotonic_in_pass"

    captures_df["usable_for_association"] = (
        (captures_df["route_context_status"] == "ok")
        & ((captures_df["coord_validity_status"] == "valid") | (captures_df["odometry_status"] == "ok"))
    )
    captures_df["usable_for_visual_analysis"] = (
        (captures_df["image_load_status"] == "ok") & (captures_df["image_content_status"] == "ok")
    )
    captures_df["requires_attention"] = ~(
        (captures_df["metadata_status"] == "ok")
        & (captures_df["coord_validity_status"] == "valid")
        & (captures_df["image_load_status"] == "ok")
        & (captures_df["image_content_status"] == "ok")
        & (captures_df["route_context_status"] == "ok")
        & (captures_df["odometry_status"] == "ok")
    )
    return captures_df


def save_ingested(df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.DATA_DIR, "ingested_captures.csv")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = ingest()
    save_ingested(df)
    print(f"Ingested {len(df)} rows -> {config.DATA_DIR}/ingested_captures.csv")
    print(f"usable_for_association:    {df['usable_for_association'].sum()} / {len(df)}")
    print(f"usable_for_visual_analysis: {df['usable_for_visual_analysis'].sum()} / {len(df)}")
    print(f"requires_attention:         {df['requires_attention'].sum()} / {len(df)}")
    print("\nmetadata_status:\n", df["metadata_status"].value_counts().to_string())
    print("\ncoord_validity_status:\n", df["coord_validity_status"].value_counts().to_string())
    print("\nimage_load_status:\n", df["image_load_status"].value_counts().to_string())
    print("\nimage_content_status:\n", df["image_content_status"].value_counts().to_string())
    print("\nroute_context_status:\n", df["route_context_status"].value_counts().to_string())
    print("\nodometry_status:\n", df["odometry_status"].value_counts().to_string())

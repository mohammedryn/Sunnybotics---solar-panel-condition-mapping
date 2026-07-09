"""
External dataset mode: adapts Sunnybotics' real sample dataset (clean/
and damaged/ image folders) into the same captures_raw schema every
downstream stage already consumes, so ingestion, condition analysis,
priority scoring, annotation, export, and visualization run UNMODIFIED
on real photos - only this module and the mode-scoping in
config.py/run_pipeline.py are new.

This is a supplemental sanity check for the visual-analysis stage
(RF-03), specifically clean vs. damaged - NOT a replacement for the
synthetic mission pipeline, which remains the full end-to-end validation
of route/odometry/panel-association/all-five-condition-categories/
edge-case handling that this dataset has no equivalent for.

Ground truth handling mirrors sim_ground_truth_internal.csv's existing
separation exactly: the folder label (clean/damaged) is written to its
own file (EXTERNAL_GROUND_TRUTH_PATH) and is NEVER merged into captures,
NEVER read by feature_extraction.py or condition_analysis.py, and is
only joined back in by external_eval_summary() at the very end, after
every inference decision has already been made from the image alone.

Known, disclosed limitations (see README):
  - Most of these real photos DO carry per-photo GPS EXIF (phone GPS at
    capture time) - extracted honestly here and written to latitude/
    longitude when present, left NaN when a given photo genuinely has
    none (never guessed at). What this dataset still does NOT have is
    route/pass structure or panel-level ground truth - no known farm
    layout to match a coordinate against - so RF-06 spatial association
    still resolves to "unresolvable" for every row, via ingest.py's
    existing schema-tolerant route-context gate (no route_pass_id/
    nominal_row_id columns here at all), not a new code path. Real GPS
    coordinates will also correctly read coord_validity_status =
    "implausible_for_farm": they describe real-world locations, which
    are naturally far outside this project's synthetic farm's arbitrary
    local coordinate frame. That's expected, not a bug. panel_row/
    panel_id are therefore still always null in the export - the E2
    columns exist, as required, but empty is the honest answer here,
    not a fabricated identity.
  - Images are resized to this project's synthetic calibration
    resolution before feature extraction, since every classical-CV
    threshold in this project was calibrated in absolute pixel units
    against that fixed canvas. This is a real, stated approximation, not
    a hidden one - a production system would need per-resolution (or
    resolution-invariant) recalibration instead.
  - timestamp/robot_id/mission_id are synthesized placeholders, clearly
    named as such (EXTERNAL-UNKNOWN, EXTERNAL-SAMPLE), never implied to
    be real captures.
"""
import os
import uuid
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import pillow_heif
from PIL import ExifTags, ImageOps
from PIL import Image as PILImage

from . import config

# Registers HEIC/HEIF with PIL's own opener so PILImage.open() handles both
# formats identically - same .getexif()/.get_ifd(GPSInfo) API, and critically
# the same ImageOps.exif_transpose() orientation handling. Without this,
# HEIC files would need a separate hand-rolled EXIF-parsing path with no
# orientation correction, which would silently analyze any rotated phone
# photo sideways - a real bug, not a hypothetical one, since 16 of the 119
# real JPGs here carry a non-1 orientation tag (cv2.imread already corrects
# those automatically; this makes HEIC match that same guarantee).
pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
# cv2.imread can't decode these without an extra codec this project doesn't
# otherwise depend on, so they get their own decode path via pillow-heif
# (see _read_image_any_format) instead of being silently skipped.
HEIC_EXTENSIONS = (".heic", ".heif")
LABEL_FOLDERS = ("clean", "damaged")

# Fixed, arbitrary namespace UUID - uuid.uuid5(namespace, name) is a
# standard, deterministic construction: the same (namespace, relative
# path) pair always produces the same UUID, on any machine, any run, no
# RNG state to manage at all. Chosen once and never changed, since
# changing it would change every external image_id.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

DEFAULT_SEED_TIME = "2026-01-01T09:00:00"
DEFAULT_CAPTURE_INTERVAL_S = 10


class ExternalDatasetNotFoundError(RuntimeError):
    pass


def _find_label_dirs(root_dir: str) -> dict:
    """Locates the clean/ and damaged/ folders anywhere under root_dir, not
    just as direct children - Sunnybotics' real sample repo nests them one
    level down (sample_images/clean, sample_images/damaged) rather than at
    the repo root, and this shouldn't require hardcoding that particular
    layout. Skips dotdirs (.git) so cloning the dataset as a git repo can't
    accidentally match anything inside it."""
    found = {}
    if not os.path.isdir(root_dir):
        return found
    for dirpath, dirnames, _ in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = os.path.basename(dirpath)
        if base in LABEL_FOLDERS and base not in found:
            found[base] = dirpath
    return found


def dataset_present(root_dir: str = None) -> bool:
    root_dir = root_dir or config.EXTERNAL_DATASET_DIR
    return bool(_find_label_dirs(root_dir))


def discover_images(root_dir: str = None) -> list:
    """Deterministically-ordered list of (absolute_path, true_label)
    tuples - sorted explicitly (not relying on os.walk/os.listdir order,
    which isn't guaranteed identical across filesystems) so image_id
    assignment and capture ordering are fully reproducible."""
    root_dir = root_dir or config.EXTERNAL_DATASET_DIR
    label_dirs = _find_label_dirs(root_dir)
    if not label_dirs:
        raise ExternalDatasetNotFoundError(
            f"External dataset not found at '{root_dir}'.\n"
            f"Clone it first:\n"
            f"  git clone {config.EXTERNAL_DATASET_CLONE_URL} {root_dir}"
        )

    records = []
    for label, folder in label_dirs.items():
        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                lower = fname.lower()
                if lower.endswith(IMAGE_EXTENSIONS) or lower.endswith(HEIC_EXTENSIONS):
                    records.append((os.path.join(dirpath, fname), label))
    records.sort(key=lambda r: r[0])

    if not records:
        raise ExternalDatasetNotFoundError(
            f"'{root_dir}' exists but no image files were found under clean/ or damaged/."
        )
    return records


def _read_image_any_format(path: str):
    """cv2.imread for anything it natively handles - it already
    auto-corrects EXIF orientation for jpg/png, confirmed directly against
    this real dataset (an orientation=6 file's loaded dimensions matched
    PIL's exif_transpose'd size, not its raw stored size). For .heic/.heif,
    PIL (with pillow-heif registered as its HEIF opener above) handles
    decode + the identical orientation correction via exif_transpose,
    then converts RGB -> BGR so both paths hand back the same ndarray
    shape everything downstream (resize, feature extraction) expects.
    Returns None on any decode failure, exactly like cv2.imread does -
    the caller's existing "processed_path unwritten -> ingest.py reports
    missing_file" path handles both identically, no new failure mode."""
    if path.lower().endswith(HEIC_EXTENSIONS):
        try:
            img = ImageOps.exif_transpose(PILImage.open(path).convert("RGB"))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imread(path)


def _dms_to_decimal(dms, ref):
    if dms is None or ref is None:
        return np.nan
    try:
        d, m, s = dms
        val = float(d) + float(m) / 60 + float(s) / 3600
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan
    return -val if ref in ("S", "W") else val


def _extract_gps(path: str):
    """Real per-photo GPS EXIF, honestly extracted - (nan, nan) when a
    photo genuinely has no GPS block (about 21% of the jpg/jpeg set) or
    when it's present but degenerate (direction ref with no lat/lon
    magnitude, seen in a handful of real files here), never guessed at.
    One decode path for every format: pillow-heif registers itself as
    PIL's HEIF opener (see module import), so PILImage.open().getexif()
    works identically for jpg/png/heic/heif - no format-specific branching
    needed here at all."""
    try:
        gps = PILImage.open(path).getexif().get_ifd(ExifTags.IFD.GPSInfo)
        if not gps:
            return np.nan, np.nan
        lat = _dms_to_decimal(gps.get(2), gps.get(1))
        lon = _dms_to_decimal(gps.get(4), gps.get(3))
        return lat, lon
    except Exception:
        return np.nan, np.nan


def _write_stub_route_and_farm():
    """Empty (headers-only) route_plan.csv / farm_truth.csv - not fabricated
    geometry, just the correct schema with zero rows. ingest.py detects
    the absence of route_pass_id/nominal_row_id columns in captures_raw
    itself and skips route/odometry checks entirely (see ingest.py); these
    stub files exist only so associate_panels.py's unconditional
    pd.read_csv() calls don't crash. Every one of associate_panels.py's
    rows will still correctly end up "unresolvable" via its EXISTING
    unprocessable-row path - no new code path in associate_panels.py."""
    route_plan_df = pd.DataFrame(columns=[
        "route_pass_id", "mission_id", "nominal_row_id",
        "start_east_m", "start_north_m", "end_east_m", "end_north_m",
        "route_direction", "expected_panel_count",
    ])
    route_plan_df.to_csv(os.path.join(config.DATA_DIR, "route_plan.csv"), index=False)

    farm_truth_df = pd.DataFrame(columns=[
        "panel_row", "panel_id", "row_idx", "panel_idx",
        "true_east_m", "true_north_m", "true_lat", "true_lon",
    ])
    farm_truth_df.to_csv(config.FARM_TRUTH_PATH, index=False)


def build_external_captures(
    root_dir: str = None,
    seed_time: str = DEFAULT_SEED_TIME,
    capture_interval_s: int = DEFAULT_CAPTURE_INTERVAL_S,
):
    """Builds captures_raw.csv (fed to the SAME ingest/associate/condition/
    priority/annotate/export pipeline as synthetic mode) plus a completely
    separate ground-truth-label file. The folder label is read here ONLY
    to route each file into `true_label` - it is never written into
    captures_raw and never touches any field an inference stage reads."""
    root_dir = root_dir or config.EXTERNAL_DATASET_DIR
    records = discover_images(root_dir)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.EXTERNAL_PROCESSED_IMAGES_DIR, exist_ok=True)
    resize_to = config.EXTERNAL_IMAGE_RESIZE_TO

    captures, ground_truth = [], []
    current_time = datetime.fromisoformat(seed_time)

    for src_path, true_label in records:
        rel_path = os.path.relpath(src_path, root_dir)
        image_id = str(uuid.uuid5(_ID_NAMESPACE, rel_path))
        processed_path = os.path.join(config.EXTERNAL_PROCESSED_IMAGES_DIR, f"{image_id}.jpg")

        img = _read_image_any_format(src_path)
        if img is not None:
            resized = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
            cv2.imwrite(processed_path, resized)
        # else: leave processed_path unwritten - ingest.py correctly reports
        # "missing_file" for a path that genuinely doesn't exist, exactly
        # like the synthetic pipeline's own missing-file edge case, with
        # no special-casing needed here.

        lat, lon = _extract_gps(src_path)
        captures.append({
            "image_id": image_id,
            "timestamp": current_time.isoformat(),
            "latitude": lat,
            "longitude": lon,
            "robot_id": "EXTERNAL-UNKNOWN",
            "mission_id": "EXTERNAL-SAMPLE",
            "image_path": processed_path,
        })
        ground_truth.append({"image_id": image_id, "true_label": true_label, "source_path": rel_path})
        current_time += timedelta(seconds=capture_interval_s)

    captures_df = pd.DataFrame(captures)
    ground_truth_df = pd.DataFrame(ground_truth)

    captures_df.to_csv(config.CAPTURES_RAW_PATH, index=False)
    ground_truth_df.to_csv(config.EXTERNAL_GROUND_TRUTH_PATH, index=False)
    _write_stub_route_and_farm()

    return captures_df, ground_truth_df


def external_eval_summary(export_df: pd.DataFrame, ground_truth_path: str = None) -> dict:
    """Joins predicted condition against the folder label ONLY here, at
    evaluation time - after every inference stage has already run. Not
    called anywhere in the inference path.

    Specifically measures the zero-shot classifier's own accuracy by reading
    `zero_shot_condition` (the original zero-shot classifier's result), not
    the primary `condition` field which after the supervised-promotion step
    holds the classifier's supervised (better-performing) call.

    clean/damaged is coarser than the 5-category classifier's own
    vocabulary (a real clean panel could still register as e.g. 'shadow'
    without the classifier being wrong - it's answering a more specific
    question than the folder split asks), so this maps `zero_shot_condition`
    to a binary clean/damaged/other call for comparison, and reports the full
    breakdown honestly rather than only a flattering top-line number."""
    ground_truth_path = ground_truth_path or config.EXTERNAL_GROUND_TRUTH_PATH
    gt = pd.read_csv(ground_truth_path)
    merged = export_df.merge(gt, on="image_id", how="left")

    def _predicted_label(condition):
        if condition == "clean":
            return "clean"
        if condition == "damage":
            return "damaged"
        return "other"

    # Read zero_shot_condition if it exists (after Task 1's primary-condition swap),
    # otherwise fall back to condition (before the swap, for backward compatibility).
    condition_column = "zero_shot_condition" if "zero_shot_condition" in merged.columns else "condition"
    merged["predicted_label"] = merged[condition_column].apply(_predicted_label)
    resolved = merged[merged["visual_analysis_status"] == "ok"]
    binary = resolved[resolved["predicted_label"].isin(["clean", "damaged"])]

    accuracy = (
        float((binary["predicted_label"] == binary["true_label"]).mean())
        if len(binary) > 0 else None
    )
    confusion = pd.crosstab(resolved["true_label"], resolved["predicted_label"]) if len(resolved) else pd.DataFrame()

    return {
        "n_total": int(len(merged)),
        "n_evaluable": int(len(resolved)),
        "n_clean_true": int((resolved["true_label"] == "clean").sum()) if len(resolved) else 0,
        "n_damaged_true": int((resolved["true_label"] == "damaged").sum()) if len(resolved) else 0,
        "n_predicted_other_category": int((resolved["predicted_label"] == "other").sum()) if len(resolved) else 0,
        "clean_vs_damaged_accuracy_on_binary_predictions": accuracy,
        "confusion_matrix": confusion.to_dict(),
        "note": (
            "External sample dataset validation only, not a production accuracy claim. "
            "'other' means the classifier predicted dirt/shadow/glare/uncertain - a real, "
            "informative result, just not a clean/damaged call the folder labels can score."
        ),
    }


def save_external_eval_summary(summary: dict, path: str = None):
    import json
    path = path or os.path.join(config.OUTPUTS_DIR, "external_eval_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def build_external_summary_chart(export_df: pd.DataFrame, ground_truth_path: str = None, path: str = None):
    """Non-spatial summary visualization - the farm grid's spatial layout
    doesn't apply here (there is no real panel geometry for this
    dataset), so this is a simple grouped bar chart instead: for each
    true label (clean/damaged), what did the zero-shot classifier actually
    predict. Directly shows the confusion pattern without pretending a spatial
    structure exists.

    Specifically visualizes the zero-shot classifier's performance by reading
    `zero_shot_condition` (the original zero-shot classifier's result), not
    the primary `condition` field which after the supervised-promotion step
    holds the classifier's supervised (better-performing) call."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ground_truth_path = ground_truth_path or config.EXTERNAL_GROUND_TRUTH_PATH
    path = path or os.path.join(config.VISUALIZATIONS_DIR, "external_summary.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    gt = pd.read_csv(ground_truth_path)
    merged = export_df.merge(gt, on="image_id", how="left")
    resolved = merged[merged["visual_analysis_status"] == "ok"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"clean": "#4caf50", "damaged": "#e53935"}
    # Read zero_shot_condition if it exists (after Task 1's primary-condition swap),
    # otherwise fall back to condition (before the swap, for backward compatibility).
    condition_column = "zero_shot_condition" if "zero_shot_condition" in resolved.columns else "condition"
    for ax, true_label in zip(axes, ["clean", "damaged"]):
        sub = resolved[resolved["true_label"] == true_label]
        counts = sub[condition_column].value_counts()
        ax.bar(counts.index, counts.values, color=colors[true_label])
        ax.set_title(f"True label: {true_label} (n={len(sub)})")
        ax.set_ylabel("predicted condition count")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

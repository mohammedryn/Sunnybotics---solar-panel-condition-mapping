"""
RF-03 support: pure per-image measurement, zero classification decisions.

Every function here answers "what physical quantity did we measure and
where" - never "what condition is this". That split is deliberate: it's
the direct answer to "how do I know you're not just recognizing your own
generator's artifacts" - every number below is a real, inspectable
optical measurement (color deviation, saturation clipping, edge geometry,
blur), computed fresh from each image against ITS OWN baseline, not
against a hardcoded template of what this project's synthetic renderer
produces.

Self-referential baseline, not an external template: each image's own
median brightness/hue is used as the "expected normal" reference, and
anomalies are measured as deviation FROM THAT, not from a stored "this is
what a clean synthetic panel looks like" comparison image. This also
means a globally warm-shifted photo (real sunlight/white-balance drift)
still gets its own consistent baseline - true localized dirt still shows
up as a hue shift RELATIVE to that image's own baseline. It does not fully
solve the white-balance problem (a real deployment would need proper
color calibration, which isn't modeled here) - stated as a limitation,
not silently assumed away.

Damage detection specifically does NOT assume a known, fixed pixel
position for the panel's grid lines. That would only work here because
this project's synthetic renderer happens to place every image at an
identical frontal perspective - hardcoding it would be indistinguishable
from "detecting this renderer," not "detecting solar panels." Instead,
the grid's own periodicity is discovered fresh per image (Hough line
detection, clustered by orientation, modal spacing found per cluster) and
any line that doesn't fit the discovered pattern is a damage candidate.
This is designed to be perspective-robust in principle - but this
dataset's images have zero perspective variation to actually stress-test
that claim against, which is disclosed rather than implied.

Every threshold constant lives in config.py and is a placeholder until
calibrated against this project's own observed score distributions (see
calibrate_thresholds.py / the README) - never picked blind.
"""
import json
import os

import cv2
import numpy as np

from . import config


def _circular_hue_diff(h1, h2):
    """OpenCV hue is 0-179 (wraps around); shortest angular distance."""
    diff = np.abs(h1.astype(np.float32) - h2)
    return np.minimum(diff, 180 - diff)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _directionality_score(mask: np.ndarray) -> float:
    """PCA eigenvalue ratio of masked pixel coordinates: ~0 for an
    isotropic blob, ~1 for a narrow directional band. This is how shadow
    (banded) is told apart from dirt (blobby) geometrically, on top of
    the color-based distinction."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0
    coords = np.stack([xs, ys]).astype(np.float32)
    cov = np.cov(coords)
    eigvals = np.linalg.eigvalsh(cov)
    lo, hi = float(eigvals[0]), float(eigvals[1])
    if hi + lo <= 1e-6:
        return 0.0
    return (hi - lo) / (hi + lo)


def _shadow_mask(hsv, v_baseline, h_baseline):
    v_drop = v_baseline.astype(np.float32) - hsv[:, :, 2].astype(np.float32)
    hue_ok = _circular_hue_diff(hsv[:, :, 0], h_baseline) < config.SHADOW_HUE_TOLERANCE
    return (v_drop > config.SHADOW_V_DROP_THRESHOLD) & hue_ok


def _dirt_mask(hsv):
    """Uses a LOCAL median-blur hue baseline, not the single global scalar
    used elsewhere - see config.py's DIRT_HUE_SHIFT_THRESHOLD comment for
    why: this project's dark base panel color makes a global hue baseline
    unstable enough to false-positive on genuinely clean images."""
    h_channel = hsv[:, :, 0]
    local_baseline = cv2.medianBlur(h_channel, config.DIRT_HUE_BASELINE_KERNEL)
    hue_shift = _circular_hue_diff(h_channel, local_baseline.astype(np.float32))
    return hue_shift > config.DIRT_HUE_SHIFT_THRESHOLD, hue_shift


def _local_variance(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f ** 2, (ksize, ksize))
    return sq_mean - mean ** 2


def _glare_mask(hsv, gray):
    near_sat = hsv[:, :, 2] > config.GLARE_NEAR_SATURATION_V
    low_texture = _local_variance(gray) < config.GLARE_LOCAL_VARIANCE_MAX
    mask = (near_sat & low_texture).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    coherent = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= config.GLARE_MIN_BLOB_AREA_PX:
            coherent[labels == label] = 1
    return coherent.astype(bool), near_sat


def _is_frame_edge(x1, y1, x2, y2, w, h, border) -> bool:
    """A line is the image's own frame only if it actually TRACES an
    edge - near-horizontal with both endpoints near the same top/bottom
    edge, or near-vertical with both endpoints near the same left/right
    edge. Deliberately not "any endpoint within `border` px of any edge":
    that cruder check was found to throw out real, genuinely dark crack
    lines that happen to run diagonally near a corner (a real image, not
    a hypothetical) - a diagonal line touching one corner is obviously
    not the rectangular frame, and shouldn't be excluded as if it were."""
    near_horizontal = abs(y1 - y2) <= border
    near_vertical = abs(x1 - x2) <= border
    top = near_horizontal and y1 < border and y2 < border
    bottom = near_horizontal and y1 > h - 1 - border and y2 > h - 1 - border
    left = near_vertical and x1 < border and x2 < border
    right = near_vertical and x1 > w - 1 - border and x2 > w - 1 - border
    return top or bottom or left or right


def _damage_lines(gray: np.ndarray, glare_mask: np.ndarray, h: int, w: int):
    """Two independent signals, computed from the same candidate line
    segments, kept separate rather than conflated:

    - `damage_line_density` (the returned mask) is now driven by absolute
      pixel DARKNESS along each line, not grid-periodicity alone. This was
      added after the periodicity-only version showed a genuine overlap
      between clean and damaged images (see README/known limitations) -
      grid-line edges dominate raw Canny output regardless of condition,
      but this project's rendered crack lines are drawn near-black
      (color (5,5,5)) while the grid's own bus-bar lines are rendered
      LIGHTER than the base fill (color (70,65,60)) - see image_synth.py.
      A line whose underlying pixels are markedly DARKER than the image's
      own baseline is a physically-motivated damage signal in the same
      self-referential spirit as the other three detectors (and matches
      how real crack/hotspot defects present as dark line anomalies in
      EL imagery), not a hardcoded template match.

      Calibration note, found by testing rather than assumed: an earlier
      pass of this exact check showed `glare` images producing the
      STRONGEST dark-line signal of all five categories - traced to (a)
      a deliberately-corrupted test file being included in the
      calibration set by mistake (excluded now, matching what
      condition_analysis.py already does at the pipeline level), and (b)
      the glare exclusion only checking each line's two ENDPOINTS against
      the glare mask, so a line could still cut through a glare blob's
      high-contrast boundary without either endpoint landing inside it.
      Fixed by excluding lines against a DILATED glare mask instead.
      Re-verified after both fixes: clean/dirt/shadow/glare all produce
      exactly zero dark-line signal, damage images range 0.0007-0.0029 -
      clean separation, not assumed.

    - `damage_irregular_line_score` (line-orientation/spacing analysis
      against the grid discovered fresh per image) is kept as a secondary,
      more diagnostic signal - still computed and reported, but no longer
      the primary detection gate, since on its own it didn't separate
      clean from damaged reliably.
    """
    edges = cv2.Canny(gray, config.DAMAGE_CANNY_LOW, config.DAMAGE_CANNY_HIGH)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=config.DAMAGE_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.DAMAGE_HOUGH_MAX_LINE_GAP,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if segments is None:
        return mask, 0, 0

    border = config.DAMAGE_BORDER_EXCLUSION_PX
    glare_dilated = cv2.dilate(glare_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
    baseline = float(np.median(gray))

    kept = []
    for (x1, y1, x2, y2) in segments[:, 0]:
        if _is_frame_edge(x1, y1, x2, y2, w, h, border):
            continue
        n = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)), 1)
        xs = np.linspace(x1, x2, n).astype(int).clip(0, w - 1)
        ys = np.linspace(y1, y2, n).astype(int).clip(0, h - 1)
        if glare_dilated[ys, xs].any():
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.hypot(x2 - x1, y2 - y1)
        perp_pos = (x1 + x2) / 2 if 45 <= angle < 135 else (y1 + y2) / 2
        darkness = baseline - float(gray[ys, xs].mean())
        kept.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "angle": angle,
            "length": length, "perp_pos": perp_pos, "darkness": darkness,
        })

    if not kept:
        return mask, 0, 0

    for seg in kept:
        if seg["darkness"] > config.DAMAGE_DARK_LINE_THRESHOLD:
            cv2.line(mask, (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]), 1, thickness=2)

    bin_deg = config.DAMAGE_ORIENTATION_BIN_DEG
    bins = {}
    for seg in kept:
        b = int(seg["angle"] // bin_deg)
        bins.setdefault(b, []).append(seg)
    dominant_bins = sorted(bins, key=lambda b: len(bins[b]), reverse=True)[:2]

    irregular = []
    regular_bins_positions = {}
    for b in dominant_bins:
        positions = sorted(seg["perp_pos"] for seg in bins[b])
        gaps = np.diff(positions)
        modal_gap = float(np.median(gaps)) if len(gaps) > 0 else None
        regular_bins_positions[b] = (positions, modal_gap)

    for seg in kept:
        b = int(seg["angle"] // bin_deg)
        if b not in dominant_bins:
            irregular.append(seg)
            continue
        positions, modal_gap = regular_bins_positions[b]
        if modal_gap is None or modal_gap == 0:
            continue
        nearest_gap = min(abs(seg["perp_pos"] - p) for p in positions if p != seg["perp_pos"]) if len(positions) > 1 else 0
        if nearest_gap > modal_gap * (1 + config.DAMAGE_SPACING_TOLERANCE_FRAC) and nearest_gap > 0:
            irregular.append(seg)

    return mask, len(irregular), len(kept)


def extract_features(image_path: str, image_id: str, save_evidence: bool = True) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    total_px = h * w

    v_baseline = np.median(hsv[:, :, 2])
    h_baseline = np.median(hsv[:, :, 0])

    shadow_mask = _shadow_mask(hsv, v_baseline, h_baseline)
    dirt_mask, dirt_hue_shift_map = _dirt_mask(hsv)
    glare_mask, glare_clipped_raw = _glare_mask(hsv, gray)
    damage_mask, n_irregular, n_total_lines = _damage_lines(gray, glare_mask, h, w)

    dirt_region_hue_shift = float(dirt_hue_shift_map[dirt_mask].mean()) if dirt_mask.any() else 0.0

    features = {
        "brightness_median": float(v_baseline) / 255.0,
        "shadow_area_ratio": float(shadow_mask.sum()) / total_px,
        "shadow_directionality_score": _directionality_score(shadow_mask),
        "dirt_area_ratio": float(dirt_mask.sum()) / total_px,
        "dirt_hue_shift_score": dirt_region_hue_shift / 90.0,  # normalize by max possible circular diff
        "glare_area_ratio": float(glare_mask.sum()) / total_px,
        "glare_clipped_pixel_ratio": float(glare_clipped_raw.sum()) / total_px,
        "damage_line_density": float(damage_mask.sum()) / total_px,
        "damage_irregular_line_score": (n_irregular / n_total_lines) if n_total_lines > 0 else 0.0,
        "laplacian_variance": _laplacian_variance(gray),
    }

    mask_paths = {}
    if save_evidence:
        os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
        for name, mask in [("shadow", shadow_mask), ("dirt", dirt_mask), ("glare", glare_mask), ("damage", damage_mask)]:
            path = os.path.join(config.EVIDENCE_DIR, f"{image_id}_{name}_mask.png")
            cv2.imwrite(path, (mask.astype(np.uint8) * 255))
            mask_paths[f"{name}_mask_path"] = path

        features_path = os.path.join(config.EVIDENCE_DIR, f"{image_id}_features.json")
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2)
        mask_paths["features_path"] = features_path

    return features, mask_paths

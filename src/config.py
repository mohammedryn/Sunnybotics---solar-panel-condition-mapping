"""
Central place for every tunable assumption in the simulation.
Change values here rather than hunting through modules - keeps the
"documented assumptions" requirement (Section 4 of the brief) auditable
from one file.
"""
from . import image_synth

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Farm geometry (fixed-tilt array, all documented assumptions) ----------
NUM_ROWS = 5
PANELS_PER_ROW = 10
PANEL_PITCH_M = 2.2      # center-to-center spacing of panels within a row
ROW_PITCH_M = 5.0        # center-to-center spacing between rows

# Arbitrary local tangent-plane anchor. Not a real Sunnybotics deployment
# site - chosen near their El Cerrito, CA office purely as a narrative
# anchor for the simulated coordinate frame.
ANCHOR_LAT = 37.9161
ANCHOR_LON = -122.3108
EARTH_RADIUS_M = 6371000.0

# Rows are modeled as running along the local "east" axis; the robot
# travels west -> east down each row (heading = 0 rad from east axis).
ROW_HEADING_RAD = 0.0

# --- GPS error model ---------------------------------------------------
# Two independent components, not i.i.d. Gaussian noise:
#  - a per-mission bias (multipath / atmospheric drift that persists for
#    the whole mission - i.e. all 5 rows/route_passes in that farm sweep,
#    not just one row)
#  - fast per-fix jitter (receiver noise, independent per capture)
# Combined magnitude lands in the brief's stated 1-3m band most of the
# time, with occasional larger excursions - that's intentional, it is
# a real edge case we want to be able to talk about in the defense.
GPS_MISSION_BIAS_STD_M = 1.0
GPS_FIX_JITTER_STD_M = 0.4

# Camera is not mounted at the GPS antenna (lever arm). Offset is in the
# robot's body frame (forward, right) and assumed known/calibrated -
# a deterministic transform, not a source of uncertainty.
LEVER_ARM_FORWARD_M = 0.3
LEVER_ARM_RIGHT_M = 0.2

# --- Odometry model ------------------------------------------------------
# Simulated wheel-encoder style along-track displacement estimate.
# Small per-step noise, no modeled long-range drift because a single row
# pass is short (~20m) - documented simplification.
ODOM_STEP_STD_M = 0.05

# --- Mission / capture timing ---------------------------------------------
MISSION_IDS = ["M01", "M02"]
MISSION_START_TIMES = {
    "M01": "2026-07-03T08:00:00",   # low sun angle -> more/longer shadows
    "M02": "2026-07-03T12:30:00",   # near solar noon -> more glare, less shadow
}
CAPTURE_INTERVAL_S = 15     # seconds between consecutive panel captures in a row
ROW_TRANSIT_S = 60          # seconds spent repositioning between rows
ROBOT_ID = "TATABOT-01"

# --- Persistent (per-panel, carries across missions) condition mix --------
PERSISTENT_CONDITION_WEIGHTS = {
    "clean": 0.60,
    "dirty": 0.28,
    "damaged": 0.12,
}

# --- Transient (per-mission, recomputed each pass) overlay probabilities ---
TRANSIENT_OVERLAY_WEIGHTS = {
    "M01": {"none": 0.60, "shadow": 0.35, "glare": 0.05},
    "M02": {"none": 0.65, "shadow": 0.10, "glare": 0.25},
}

# --- Corrupted / unusable capture injection --------------------------------
# Independent of panel condition - simulates a real capture failure
# (lens flare whiteout, motion blur, sensor glitch). Exercises RF-01
# crash-resistance and the mandatory `uncertain` category.
CORRUPTED_CAPTURE_RATE = 0.03

# --- Association ambiguity (used later by associate_panels.py) ------------
ASSOCIATION_CONFIDENCE_MATCHED_THRESHOLD = 0.7

# --- Ingestion validation thresholds (src/ingest.py) -----------------------
# Below this pixel std, an image is considered blank/near-constant.
# SIMULATION-SPECIFIC: calibrated empirically against this project's own
# procedurally-rendered images (normal ~11.5-13.5, blackout/overexposed
# ~0-0.3, heavy blur ~8.9). Real photographs have a much wider and
# scene-dependent std distribution - this threshold would need
# recalibration against real panel photos before reuse on real data.
BLANK_STD_THRESHOLD = 5.0
MIN_IMAGE_DIMENSION_PX = 10

# Tolerance (meters) around [0, route_length_m] for odom_cumulative_m
# plausibility checks - allows for the small measurement noise already
# present in the simulated odometry (observed up to ~0.07m negative
# excursion on first-in-pass readings) without flagging it as an
# impossible value, while still catching genuinely broken readings.
ROUTE_ODOM_TOLERANCE_M = 1.0

# --- Association (src/associate_panels.py, RF-06) --------------------------
# Uncertainty floor for computing panel-posterior CONFIDENCE (not for the
# Kalman filter's own internal recursion, which stays mathematically
# correct). Prevents repeated GPS updates from shrinking reported
# confidence toward false certainty - a real KF is "consistent" only if
# its reported covariance actually bounds the true error, and an
# unmodeled bias means raw P_est understates that error after several
# updates.
ASSOCIATION_VARIANCE_FLOOR_M2 = 0.35 ** 2

# How much smaller top-2's posterior must be than top-1's before the
# result counts as "matched" rather than "ambiguous" - a Lowe's-ratio-test
# analogue for spatial candidates, not just a raw top-1 threshold.
# MUST exceed (2*ASSOCIATION_CONFIDENCE_MATCHED_THRESHOLD - 1): for any
# discrete probability distribution, top2 <= 1-top1 (it's at most all
# remaining mass), so margin = top1-top2 >= 2*top1-1 whenever the
# confidence gate has already passed. At threshold 0.70 that bound is
# 0.4 - a margin threshold at or below that is unreachable dead code,
# since passing the confidence gate already guarantees a bigger margin.
# Set above the bound so this is a real, additional, stricter gate.
TOP2_MARGIN_MATCHED_THRESHOLD = 0.45

# Mean GPS-vs-odometry innovation (meters) across a route pass beyond
# which a persistent bias (not just per-fix jitter) is suspected - tied
# to the point where it would actually start flipping panel assignment.
BIAS_INNOVATION_THRESHOLD_M = PANEL_PITCH_M / 2

# Cross-track distance (meters) from the nominal row's centerline beyond
# which a GPS reading is "strongly" inconsistent with the claimed row -
# equal to the Voronoi boundary between adjacent rows.
CROSS_TRACK_STRONG_MISMATCH_M = ROW_PITCH_M / 2

# Soft prior variance for a pass's very first capture (route start is
# known by construction; this represents residual uncertainty about
# exactly where within that a capture happened before any delta/GPS
# is incorporated).
ROUTE_START_PRIOR_VARIANCE_M2 = 0.1 ** 2

# Inflated process variance applied for one step when odometry_status
# is not "ok" - we don't trust the reported delta, so we assume zero
# motion with a "we genuinely don't know" uncertainty of about one
# panel-pitch, rather than propagating a wrong number forward.
ODOM_REJECTED_STEP_VARIANCE_M2 = PANEL_PITCH_M ** 2

# --- Condition analysis (src/feature_extraction.py, src/condition_analysis.py, RF-03) ---
# All thresholds below are PLACEHOLDERS pending empirical calibration
# against this project's own feature-score distributions (same discipline
# as every other threshold in this project - never picked blind).
EVIDENCE_DIR = "outputs/evidence"

BLUR_LAPLACIAN_VARIANCE_THRESHOLD = 50.0  # below this, image ruled "unusable" (too blurry)

# Calibrated against this project's own images: at thresh=20 the shadow
# detector badly under-fired (max ~0.001 area even on true shadow images);
# at thresh=10, clean/dirt/damage top out at ~0.018 while shadow images
# average ~0.11 - a solid margin. See feature_extraction.py's docstring
# for why the V-channel didn't need the same local-baseline fix hue did.
SHADOW_V_DROP_THRESHOLD = 10
SHADOW_HUE_TOLERANCE = 10     # max circular hue deviation (0-179 scale) still called "same surface"

# A single global hue baseline caused false dirt-positives on genuinely
# CLEAN images (~0.23 area ratio, nearly matching real dirt's ~0.30) -
# root cause: this project's dark base panel color (V~40/255) makes hue
# numerically unstable under per-cell rendering variation and sensor
# noise (small absolute BGR differences = large relative/hue swings at
# low brightness). Fixed by comparing each pixel to a LOCAL median-blur
# baseline (kernel below) rather than one global scalar, so gradual
# cell-to-cell variation reads as expected while a genuinely localized
# dirt blotch still stands out against its own neighborhood. Calibrated:
# kernel=31/threshold=12 separates clean (max 0.011) from dirt (min
# 0.038) with a healthy margin - see README for the verification.
DIRT_HUE_BASELINE_KERNEL = 31
DIRT_HUE_SHIFT_THRESHOLD = 12

GLARE_NEAR_SATURATION_V = 235       # V-channel threshold for "near-clipped"
GLARE_LOCAL_VARIANCE_MAX = 15.0     # blown-out patches have low internal texture variance
GLARE_MIN_BLOB_AREA_PX = 30

DAMAGE_CANNY_LOW, DAMAGE_CANNY_HIGH = 50, 150
DAMAGE_HOUGH_MIN_LINE_LENGTH = 15
DAMAGE_HOUGH_MAX_LINE_GAP = 5
DAMAGE_BORDER_EXCLUSION_PX = 4       # ignore lines this close to the image edge (frame, not panel)
DAMAGE_ORIENTATION_BIN_DEG = 10      # bin width for clustering line orientations
DAMAGE_SPACING_TOLERANCE_FRAC = 0.35  # fractional deviation from modal spacing before "irregular"

# Grayscale points darker than the image's own median before a candidate
# line counts as a damage signal (see feature_extraction.py's
# _damage_lines docstring for the full story). Calibrated against this
# project's own data after excluding a corrupted-file confound and fixing
# a glare-exclusion gap: clean/dirt/shadow/glare all measured exactly 0.0
# at this threshold, damage images ranged 0.0007-0.0029 - verified
# separation, not picked blind.
DAMAGE_DARK_LINE_THRESHOLD = 15

# Per-issue detection thresholds on the raw area_ratio/density features,
# calibrated against this project's own observed score distributions
# (see README for the full calibration table):
#   dirt_area_ratio:     clean max=0.0148, dirt min=0.0328   -> 0.020
#   shadow_area_ratio:   clean max=0.0067, shadow min~0      -> 0.010
#                        (some genuinely weak shadows will be missed -
#                        disclosed, not silently accepted)
#   glare_area_ratio:    clean/other max=0.0000, glare min=0.026 -> 0.010
#   damage_line_density: originally a pure grid-periodicity signal, which
#                        genuinely overlapped between clean and damaged
#                        images (clean max=0.0214, damage min=0.0144) -
#                        even a fully clean synthetic image has ~10,500+
#                        Canny edge pixels from the grid structure alone,
#                        and segment-length filtering was tested and did
#                        not help separate them. Replaced with a
#                        darkness-based signal (see feature_extraction.py's
#                        _damage_lines docstring) that measures whether
#                        candidate line pixels are markedly darker than
#                        the image's own baseline - real crack lines are
#                        rendered near-black, bus-bar grid lines are
#                        rendered lighter than the base fill, so this
#                        distinguishes them on a physically real basis
#                        rather than geometry alone. On the new scale,
#                        clean/dirt/shadow/glare all measure exactly 0.0,
#                        damage images range 0.0007-0.0029 -> threshold
#                        set well inside that gap, not against the old
#                        scale.
ISSUE_DETECTION_THRESHOLDS = {
    "dirt": 0.020,
    "shadow": 0.010,
    "glare": 0.010,
    "damage": 0.0004,
}
CONDITION_TOP2_MARGIN_THRESHOLD = 0.15  # below this margin between top-1/top-2 issue confidence -> uncertain

# Temporal cross-check: only trust a predicted_panel_id pairing across
# missions when BOTH images independently got a confident ("matched")
# association - not "ambiguous or better", specifically to avoid
# reinforcing an association error via a wrongly-grouped pair.
TEMPORAL_CROSSCHECK_REQUIRED_STATUS = "matched"

# --- Priority scoring (src/priority_score.py, RF-04) ------------------------
# Band boundaries are the brief's own spec, not invented: 0-20 low, 21-50
# medium, 51-80 high, 81-100 critical.
PRIORITY_BAND_BOUNDS = {"low": 20, "medium": 50, "high": 80}  # critical = above 80

# Damage floors HIGH regardless of confidence - deliberately asymmetric:
# a missed real crack is expensive (equipment/fire risk, further
# degradation); a false-positive inspection dispatch just costs one
# extra visit. Confidence only pushes further into "critical", never
# below "high".
DAMAGE_PRIORITY_FLOOR = 70.0
DAMAGE_PRIORITY_CONFIDENCE_RANGE = 30.0  # floor + this*confidence, capped at 100

# Dirt scales continuously with MEASURED AREA (not confidence) - cleaning
# urgency should track how much of the panel is actually soiled (drives
# energy loss), not how sure the classifier is that it's dirt.
DIRT_PRIORITY_MIN = 20.0
DIRT_PRIORITY_MAX = 75.0
DIRT_SEVERITY_SATURATION_RATIO = 0.06  # ~3x the detection threshold (0.02)

# Shadow/glare: deliberately low-to-medium and flat - usually transient,
# not worth a cleaning dispatch, but non-zero since the panel underneath
# genuinely couldn't be assessed.
SHADOW_GLARE_PRIORITY_SCORE = 30.0

# Uncertain: moderate, never "clean" - ambiguous evidence prompts a
# look, not an assumed cleaning dispatch.
UNCERTAIN_PRIORITY_SCORE = 35.0

CLEAN_PRIORITY_SCORE = 5.0

# Unusable images bypass condition logic entirely - this is an
# operations problem (no data), not a panel-condition problem.
UNUSABLE_RECAPTURE_PRIORITY_SCORE = 70.0

# --- Paths, mode-scoped (src/config.py, pipeline modes) --------------------
# Two pipeline modes share every downstream module (ingest, associate,
# condition analysis, priority, annotate, export, visualize):
#   synthetic - the full simulated dataset (GPS, route, odometry, panel
#               association, all 5 condition categories, injected edge
#               cases). This is the main, full-system pipeline.
#   external  - Sunnybotics' real sample dataset (clean/ and damaged/
#               image folders only, no GPS/route/panel metadata). A
#               supplemental sanity check for the visual-analysis stage
#               specifically, not a replacement for the synthetic
#               end-to-end validation - see src/external_dataset.py.
#
# All path constants below are DERIVED from MODE and are mutated by
# set_mode() - every module that uses one of these as a function
# parameter default MUST look it up at call time (`path = path or
# config.X`), never bind it directly in the signature
# (`def f(path=config.X)`), since signature defaults are evaluated once
# at import time, before set_mode() can possibly run. This was audited
# and fixed across every module when mode support was added - if you add
# a new function that reads one of these paths, follow the same pattern.
MODE = "synthetic"


def set_mode(mode: str):
    global MODE, DATA_DIR, RAW_IMAGES_DIR, FARM_TRUTH_PATH, CAPTURES_RAW_PATH
    global SIM_GROUND_TRUTH_PATH, OUTPUTS_DIR, ANNOTATED_DIR, VISUALIZATIONS_DIR, EVIDENCE_DIR
    if mode not in ("synthetic", "external"):
        raise ValueError(f"unknown mode {mode!r}, expected 'synthetic' or 'external'")
    MODE = mode
    DATA_DIR = f"data/{mode}"
    RAW_IMAGES_DIR = f"{DATA_DIR}/raw_images"
    FARM_TRUTH_PATH = f"{DATA_DIR}/farm_truth.csv"
    CAPTURES_RAW_PATH = f"{DATA_DIR}/captures_raw.csv"
    SIM_GROUND_TRUTH_PATH = f"{DATA_DIR}/sim_ground_truth_internal.csv"
    OUTPUTS_DIR = f"outputs/{mode}"
    ANNOTATED_DIR = f"{OUTPUTS_DIR}/annotated"
    VISUALIZATIONS_DIR = f"{OUTPUTS_DIR}/visualizations"
    EVIDENCE_DIR = f"{OUTPUTS_DIR}/evidence"


set_mode("synthetic")  # establish default paths at import time

# --- External dataset (src/external_dataset.py) -----------------------
# Sunnybotics' real sample dataset, provided mid-assignment: two image
# folders (clean/, damaged/), no GPS/route/panel metadata. Expected local
# location - not committed to this repo (fetch it yourself, see README).
EXTERNAL_DATASET_DIR = "data/external/sunnybotics-solar-panel-challenge"
EXTERNAL_DATASET_CLONE_URL = "https://github.com/roboticsSunnyApp/sunnybotics-solar-panel-challenge"
EXTERNAL_PROCESSED_IMAGES_DIR = "data/external/processed_images"
EXTERNAL_GROUND_TRUTH_PATH = "data/external/external_ground_truth_internal.csv"
# External photos arrive at arbitrary real-world resolutions. Every
# classical-CV threshold in this project (edge counts, blob areas, line
# lengths in ISSUE_DETECTION_THRESHOLDS etc.) was calibrated in absolute
# pixel units against the synthetic renderer's fixed canvas - resizing
# external images to match is what makes those thresholds meaningful for
# them at all, not a cosmetic step. Disclosed as a real limitation in the
# README, not hidden: a production system would need per-resolution (or
# resolution-invariant) recalibration instead.
EXTERNAL_IMAGE_RESIZE_TO = (image_synth.IMG_W, image_synth.IMG_H)

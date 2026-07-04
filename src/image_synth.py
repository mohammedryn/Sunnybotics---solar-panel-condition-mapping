"""
Procedural solar-panel image generation (RF-01 support; also produces
the ground-truth labels condition analysis will later be scored against).

No dataset link was provided for this assignment. Rather than sourcing
an uncertain-license public dataset, images are generated procedurally
so that:
  - every category (clean/dirty/shadowed/glare/damaged/uncertain) has
    guaranteed, controlled coverage
  - there is zero license risk
  - we have real ground truth to validate the condition-analysis and
    association pipelines against later

Documented limitation (stated again in the technical report): these are
stylised renders, not photographs. They capture the illumination/color
phenomena real heuristics key off (directional darkening for shadow,
color-cast blotching for dirt, blown highlights for glare) but not the
full texture complexity of real panel photos. Classification validity
demonstrated on this data is simulation-only, not a real-world accuracy
claim.
"""
import os

import cv2
import numpy as np

IMG_W, IMG_H = 480, 360
CELL_COLS, CELL_ROWS = 6, 10


def _base_cell_grid(rng: np.random.Generator) -> np.ndarray:
    """Dark PV-module base texture: a grid of cells with thin bus-bar
    lines and mild per-cell brightness variation, plus sensor noise."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    base_color = np.array([40, 30, 20], dtype=np.float32)  # BGR, dark navy/near-black
    img[:] = base_color

    cell_w = IMG_W / CELL_COLS
    cell_h = IMG_H / CELL_ROWS
    for r in range(CELL_ROWS):
        for c in range(CELL_COLS):
            variation = rng.normal(0, 4, size=3)
            y0, y1 = int(r * cell_h) + 2, int((r + 1) * cell_h) - 2
            x0, x1 = int(c * cell_w) + 2, int((c + 1) * cell_w) - 2
            img[y0:y1, x0:x1] += variation

    # bus-bar grid lines
    line_color = np.array([70, 65, 60], dtype=np.float32)
    for r in range(CELL_ROWS + 1):
        y = min(int(r * cell_h), IMG_H - 1)
        img[max(y - 1, 0):y + 1, :] = line_color
    for c in range(CELL_COLS + 1):
        x = min(int(c * cell_w), IMG_W - 1)
        img[:, max(x - 1, 0):x + 1] = line_color

    noise = rng.normal(0, 2.5, size=img.shape)
    img += noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_dirty(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brownish/grey blotches + reduced local contrast - dust and soiling."""
    overlay = img.copy().astype(np.float32)
    n_blotches = rng.integers(6, 14)
    for _ in range(n_blotches):
        cx, cy = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        axes = (int(rng.integers(15, 45)), int(rng.integers(10, 30)))
        angle = rng.integers(0, 180)
        color = (30, 55, 70)  # BGR, dusty brown-grey
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        alpha = rng.uniform(0.25, 0.45)
        for ch in range(3):
            overlay[:, :, ch] = np.where(mask == 255,
                                          overlay[:, :, ch] * (1 - alpha) + color[ch] * alpha,
                                          overlay[:, :, ch])
    overlay = overlay * 0.9 + 15  # flatten contrast slightly (dust scatters light)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_shadow(img: np.ndarray, rng: np.random.Generator, strong: bool) -> np.ndarray:
    """Directional darkening band - simulates a structure/adjacent-row
    shadow. `strong` = low sun angle (mission M01), wider/darker band."""
    overlay = img.astype(np.float32).copy()
    angle_deg = rng.uniform(20, 60) if strong else rng.uniform(60, 85)
    width_frac = rng.uniform(0.45, 0.75) if strong else rng.uniform(0.15, 0.35)
    darkness = rng.uniform(0.35, 0.55) if strong else rng.uniform(0.15, 0.3)

    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    band_w = int(IMG_W * width_frac)
    x_start = rng.integers(-band_w // 2, IMG_W)
    pts = np.array([
        [x_start, 0], [x_start + band_w, 0],
        [x_start + band_w + int(IMG_H / np.tan(np.radians(max(angle_deg, 1)))), IMG_H],
        [x_start + int(IMG_H / np.tan(np.radians(max(angle_deg, 1)))), IMG_H],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    factor = 1.0 - (mask.astype(np.float32) / 255.0) * darkness
    for ch in range(3):
        overlay[:, :, ch] *= factor
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blown-out specular highlight blobs - direct reflection near solar noon."""
    overlay = img.astype(np.float32).copy()
    n_blobs = rng.integers(1, 3)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        radius = int(rng.integers(30, 80))
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        alpha = mask.astype(np.float32) / 255.0
        for ch in range(3):
            overlay[:, :, ch] = overlay[:, :, ch] * (1 - alpha) + 255 * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_damage(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crack lines + a discoloration patch (delamination/hotspot proxy)."""
    overlay = img.copy()
    n_cracks = rng.integers(2, 5)
    for _ in range(n_cracks):
        x, y = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        pts = [(x, y)]
        for _ in range(rng.integers(3, 6)):
            x = int(np.clip(x + rng.integers(-40, 40), 0, IMG_W - 1))
            y = int(np.clip(y + rng.integers(-40, 40), 0, IMG_H - 1))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], (5, 5, 5), thickness=rng.integers(1, 3))

    cx, cy = rng.integers(50, IMG_W - 50), rng.integers(50, IMG_H - 50)
    axes = (int(rng.integers(20, 40)), int(rng.integers(15, 30)))
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, rng.integers(0, 180), 0, 360, 255, -1)
    overlay = overlay.astype(np.float32)
    color = (25, 60, 90)  # yellow-brown discoloration, BGR
    alpha = 0.4
    for ch in range(3):
        overlay[:, :, ch] = np.where(mask == 255,
                                      overlay[:, :, ch] * (1 - alpha) + color[ch] * alpha,
                                      overlay[:, :, ch])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_corruption(img: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "heavy_blur":
        return cv2.GaussianBlur(img, (35, 35), 0)
    if mode == "blackout":
        return np.clip(img.astype(np.float32) * 0.02, 0, 255).astype(np.uint8)
    if mode == "overexposed":
        return np.full_like(img, 250)
    raise ValueError(f"unknown corruption mode: {mode}")


def synthesize_capture_image(
    rng: np.random.Generator,
    persistent_condition: str,
    transient_overlay: str,
    mission_id: str,
    is_corrupted: bool = False,
    corruption_mode: str = None,
):
    """Renders one capture. Returns (image_bgr, detected_issues list)."""
    img = _base_cell_grid(rng)
    issues = []

    if persistent_condition == "dirty":
        img = _apply_dirty(img, rng)
        issues.append("dirt")
    elif persistent_condition == "damaged":
        img = _apply_damage(img, rng)
        issues.append("damage")

    if transient_overlay == "shadow":
        img = _apply_shadow(img, rng, strong=(mission_id == "M01"))
        issues.append("shadow")
    elif transient_overlay == "glare":
        img = _apply_glare(img, rng)
        issues.append("glare")

    if is_corrupted:
        img = _apply_corruption(img, rng, corruption_mode)
        issues.append("corrupted")

    return img, issues


def save_image(img: np.ndarray, path: str, truncate: bool = False, rng: np.random.Generator = None):
    """Writes the image. `truncate=True` deliberately corrupts the file
    on disk afterwards (simulates a real corrupted-transfer JPEG) so
    RF-01's crash-resistance is exercised against a genuinely broken file,
    not just a visually-degraded-but-valid one."""
    cv2.imwrite(path, img)
    if truncate:
        with open(path, "rb") as f:
            data = f.read()
        cut = int(len(data) * rng.uniform(0.2, 0.5))
        with open(path, "wb") as f:
            f.write(data[:cut])

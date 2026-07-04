"""
RF-05: annotated image generation.

Every processed image gets an annotated version in outputs/annotated/,
reusing the traceability chain rather than re-deriving anything: the
evidence mask tinted onto the image comes directly from
feature_extraction.py's saved masks (Section 3), the panel identity from
associate_panels.py's predicted_panel_id (Section 2, RF-06 - never
ground truth), and the priority/action from priority_score.py (Section 4).

For a genuinely missing/corrupted image there is no base photo to
annotate - RF-05 still requires an output for every processed image, so
a placeholder canvas is rendered with a bold warning instead of silently
skipping it.
"""
import os

import cv2
import numpy as np
import pandas as pd

from . import config, image_synth

MASK_TINT_COLORS = {  # BGR
    "dirt": (30, 90, 140),
    "shadow": (140, 60, 20),
    "glare": (20, 200, 230),
    "damage": (20, 20, 200),
}
BAND_COLORS = {  # BGR, traffic-light scheme
    "low": (60, 170, 60),
    "medium": (30, 200, 230),
    "high": (0, 140, 255),
    "critical": (0, 0, 220),
}


def _load_mask(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _tint_and_box(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> np.ndarray:
    out = img.copy().astype(np.float32)
    mask_bool = mask > 0
    for c in range(3):
        out[:, :, c] = np.where(mask_bool, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    out = out.astype(np.uint8)

    ys, xs = np.nonzero(mask_bool)
    if len(xs) > 0:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
    return out


def _placeholder_canvas(message: str) -> np.ndarray:
    canvas = np.full((image_synth.IMG_H, image_synth.IMG_W, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (2, 2), (image_synth.IMG_W - 3, image_synth.IMG_H - 3), (0, 0, 200), 3)
    _put_text_block(canvas, [message], origin=(15, image_synth.IMG_H // 2 - 10), color=(0, 0, 220), scale=0.55)
    return canvas


def _put_text_block(img, lines, origin=(8, 18), color=(255, 255, 255), scale=0.45, line_height=18):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_height


def annotate_one(row: pd.Series) -> np.ndarray:
    if row["visual_analysis_status"] in ("image_missing", "image_corrupt"):
        return _placeholder_canvas(f"{row['visual_analysis_status'].upper()} - RECAPTURE REQUIRED")

    img = cv2.imread(row["image_path"])
    if img is None:
        return _placeholder_canvas("IMAGE UNREADABLE - RECAPTURE REQUIRED")

    if row["visual_analysis_status"] == "unusable":
        _put_text_block(img, [f"UNUSABLE: {row['unusable_reason']}", "Priority: recapture"],
                         origin=(10, 24), color=(0, 0, 230), scale=0.55)
        return img

    condition = row["condition"]
    if condition in MASK_TINT_COLORS and isinstance(row.get("evidence_bbox_or_mask_path"), str):
        mask = _load_mask(row["evidence_bbox_or_mask_path"])
        if mask is not None:
            img = _tint_and_box(img, mask, MASK_TINT_COLORS[condition])

    band_color = BAND_COLORS.get(row["priority_band"], (255, 255, 255))
    lines = [
        f"Condition: {condition} ({row['condition_confidence']:.2f})" if pd.notna(row["condition_confidence"])
        else f"Condition: {condition}",
        f"Priority: {row['cleaning_priority_score']:.0f} ({row['priority_band']}) -> {row['recommended_action']}",
    ]
    if row.get("predicted_panel_id") and row.get("association_status") == "matched":
        lines.append(f"Panel: {row['predicted_panel_id']}")
    elif row.get("predicted_panel_id"):
        lines.append(f"Panel: {row['predicted_panel_id']} (WARNING: {row['association_status']})")
    else:
        lines.append(f"WARNING: panel unresolved ({row.get('association_status')})")

    _put_text_block(img, lines[:2], origin=(8, 18), color=(255, 255, 255), scale=0.42)
    _put_text_block(img, [lines[2]], origin=(8, 18 + 2 * 16 + 4), color=band_color, scale=0.42)
    return img


def annotate_all(joined_df: pd.DataFrame, output_dir: str = None) -> dict:
    output_dir = output_dir or config.ANNOTATED_DIR
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for _, row in joined_df.iterrows():
        img = annotate_one(row)
        path = os.path.join(output_dir, f"{row['image_id']}.jpg")
        cv2.imwrite(path, img)
        paths[row["image_id"]] = path
    return paths

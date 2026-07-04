"""
RF-07 (+bonus): farm grid visualization.

A re-projected lat/lon map over this project's fictional anchor point
would be less honest and less useful than showing the actual row/panel
structure directly - see the README for the full reasoning. Two grids
side by side (one per mission) since every panel gets one capture per
mission; showing both makes the persistent-vs-transient story (Section 1
and 3's whole reason for the two-mission design) visible spatially too.

Cells are placed at each image's PREDICTED panel position (from
association, RF-06) - never ground truth. An image with no resolved
panel (association_status in unresolvable/out_of_bounds) cannot be
placed on the grid at all and is listed separately rather than silently
dropped or guessed onto a cell.

Two outputs, under outputs/visualizations/:
  farm_grid.png            - static, always reproducible
  farm_grid_interactive.html - self-contained (no server needed, unlike
                                Streamlit - keeps the brief's "1-2
                                command" requirement), dropdown to
                                switch coloring, hover tooltips with
                                full detail
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import plotly.graph_objects as go

from . import config

ACTION_COLORS = {
    "none": "#4caf50", "clean": "#ff9800", "inspect": "#e53935",
    "human_review": "#9c27b0", "recapture": "#757575", None: "#cfcfcf",
}
BAND_COLORS = {
    "low": "#4caf50", "medium": "#ffc107", "high": "#ff9800", "critical": "#e53935", None: "#cfcfcf",
}
CONDITION_COLORS = {
    "clean": "#4caf50", "dirt": "#8d6e63", "shadow": "#1976d2", "glare": "#fbc02d",
    "damage": "#e53935", "uncertain": "#9e9e9e", None: "#cfcfcf",
}
ASSOC_COLORS = {
    "matched": "#4caf50", "ambiguous": "#ff9800", "unresolvable": "#e53935",
    "out_of_bounds": "#9c27b0", None: "#cfcfcf",
}
COLOR_MAPS = {
    "recommended_action": ACTION_COLORS, "priority_band": BAND_COLORS,
    "condition": CONDITION_COLORS, "association_status": ASSOC_COLORS,
}

_PANEL_ID_RE = re.compile(r"R(\d+)-P(\d+)")


def _panel_position(panel_id):
    if not isinstance(panel_id, str):
        return None
    m = _PANEL_ID_RE.match(panel_id)
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2)) - 1  # 0-indexed row, panel


def static_farm_grid(export_df: pd.DataFrame, color_by: str = "recommended_action", path: str = None):
    path = path or os.path.join(config.VISUALIZATIONS_DIR, "farm_grid.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    missions = sorted(export_df["mission_id"].dropna().unique())
    color_map = COLOR_MAPS[color_by]

    fig, axes = plt.subplots(1, len(missions), figsize=(7 * len(missions), 4.5))
    if len(missions) == 1:
        axes = [axes]

    for ax, mission_id in zip(axes, missions):
        ax.set_title(f"Mission {mission_id} - colored by {color_by}")
        sub = export_df[export_df["mission_id"] == mission_id]
        placed = 0
        for _, row in sub.iterrows():
            pos = _panel_position(row["panel_id"])
            if pos is None:
                continue
            r, c = pos
            placed += 1
            color = color_map.get(row[color_by], "#cfcfcf")
            ax.add_patch(mpatches.Rectangle((c, config.NUM_ROWS - 1 - r), 0.96, 0.96, facecolor=color, edgecolor="black"))
            ax.text(c + 0.48, config.NUM_ROWS - 1 - r + 0.62, str(row["panel_id"]).replace("R", "").replace("-P", "."),
                     ha="center", va="center", fontsize=7)
            ax.text(c + 0.48, config.NUM_ROWS - 1 - r + 0.30, str(row["condition"]),
                     ha="center", va="center", fontsize=6)

        ax.set_xlim(0, config.PANELS_PER_ROW)
        ax.set_ylim(0, config.NUM_ROWS)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        unresolved = len(sub) - placed
        if unresolved:
            ax.set_xlabel(f"{unresolved} image(s) with no resolved panel position (not shown)")

    handles = [mpatches.Patch(color=v, label=str(k)) for k, v in color_map.items() if k is not None]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def interactive_farm_grid(export_df: pd.DataFrame, path: str = None):
    path = path or os.path.join(config.VISUALIZATIONS_DIR, "farm_grid_interactive.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    missions = sorted(export_df["mission_id"].dropna().unique())
    color_fields = list(COLOR_MAPS.keys())

    fig = go.Figure()
    n_traces_per_field = []

    for field in color_fields:
        color_map = COLOR_MAPS[field]
        n_traces_this_field = 0
        for mission_id in missions:
            sub = export_df[export_df["mission_id"] == mission_id].dropna(subset=["panel_id"])
            positions = sub["panel_id"].map(_panel_position)
            valid = positions.notna()
            sub, positions = sub[valid], positions[valid]
            if len(sub) == 0:
                continue
            xs = [p[1] for p in positions]
            ys = [config.NUM_ROWS - 1 - p[0] for p in positions]
            colors = [color_map.get(v, "#cfcfcf") for v in sub[field]]
            hover = [
                f"panel_id={r.panel_id}<br>condition={r.condition} ({r.confidence:.2f})<br>"
                f"priority={r.cleaning_priority_score:.0f} ({r.priority_band}) -> {r.recommended_action}<br>"
                f"association={r.association_status} ({r.association_confidence})<br>"
                f"visual_status={r.visual_analysis_status}"
                for r in sub.itertuples()
            ]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(size=38, color=colors, symbol="square", line=dict(width=1, color="black")),
                text=[str(pid) for pid in sub["panel_id"]], textposition="middle center", textfont=dict(size=8),
                hovertext=hover, hoverinfo="text",
                name=f"{mission_id}", visible=(field == color_fields[0]),
                xaxis="x" if mission_id == missions[0] else "x2",
                yaxis="y" if mission_id == missions[0] else "y2",
            ))
            n_traces_this_field += 1
        n_traces_per_field.append(n_traces_this_field)

    buttons = []
    trace_idx = 0
    for field, n in zip(color_fields, n_traces_per_field):
        visible = [False] * sum(n_traces_per_field)
        for i in range(trace_idx, trace_idx + n):
            visible[i] = True
        buttons.append(dict(label=f"Color by: {field}", method="update", args=[{"visible": visible}]))
        trace_idx += n

    fig.update_layout(
        title="Farm grid - traceability view (hover a panel for full detail)",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.15)],
        xaxis=dict(domain=[0, 0.48], title=str(missions[0]) if missions else "", showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis2=dict(domain=[0.52, 1.0], title=str(missions[1]) if len(missions) > 1 else "", showgrid=False, zeroline=False),
        yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False, anchor="x2"),
        height=500, width=1100,
    )
    fig.write_html(path)
    return path

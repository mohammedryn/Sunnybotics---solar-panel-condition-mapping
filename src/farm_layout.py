"""
Ground-truth farm geometry (RF-01/RF-02 support).

Everything here is the *known map* - the physical layout of panels the
association algorithm (RF-06) reconciles noisy GPS against. Nothing in
this module is exported into captures_raw or any RF-02 field; it is the
map, not a measurement.

Coordinates are generated in a local ENU (East-North-Up) tangent-plane
frame centered on an arbitrary anchor point, then converted to lat/lon
for anything that needs to look like real sensor output. Reasoning in
meters on a flat local plane avoids the distortion and precision loss of
doing spatial math directly in lat/lon degrees, and matches how a real
robotics stack (e.g. an EKF localization filter) would represent position.

The equirectangular approximation used for ENU<->lat/lon is only valid
at small scale (this farm spans under 100m) - it would need a proper
projection (e.g. UTM) for a farm spanning kilometers.
"""
import numpy as np
import pandas as pd

from . import config


def enu_to_latlon(east_m, north_m, anchor_lat=config.ANCHOR_LAT, anchor_lon=config.ANCHOR_LON):
    lat = anchor_lat + (north_m / config.EARTH_RADIUS_M) * (180.0 / np.pi)
    lon = anchor_lon + (east_m / (config.EARTH_RADIUS_M * np.cos(np.radians(anchor_lat)))) * (180.0 / np.pi)
    return lat, lon


def latlon_to_enu(lat, lon, anchor_lat=config.ANCHOR_LAT, anchor_lon=config.ANCHOR_LON):
    north_m = (lat - anchor_lat) * (np.pi / 180.0) * config.EARTH_RADIUS_M
    east_m = (lon - anchor_lon) * (np.pi / 180.0) * config.EARTH_RADIUS_M * np.cos(np.radians(anchor_lat))
    return east_m, north_m


def generate_farm(
    num_rows=config.NUM_ROWS,
    panels_per_row=config.PANELS_PER_ROW,
    panel_pitch_m=config.PANEL_PITCH_M,
    row_pitch_m=config.ROW_PITCH_M,
) -> pd.DataFrame:
    """Ground-truth panel grid. One row per panel."""
    records = []
    for r in range(num_rows):
        for p in range(panels_per_row):
            east_m = p * panel_pitch_m
            north_m = r * row_pitch_m
            lat, lon = enu_to_latlon(east_m, north_m)
            records.append(
                {
                    "panel_row": f"R{r + 1}",
                    "panel_id": f"R{r + 1}-P{p + 1:02d}",
                    "row_idx": r,
                    "panel_idx": p,
                    "true_east_m": east_m,
                    "true_north_m": north_m,
                    "true_lat": lat,
                    "true_lon": lon,
                }
            )
    return pd.DataFrame(records)


def save_farm_truth(df: pd.DataFrame, path: str = config.FARM_TRUTH_PATH):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    farm = generate_farm()
    save_farm_truth(farm)
    print(f"Generated {len(farm)} panels across {config.NUM_ROWS} rows -> {config.FARM_TRUTH_PATH}")

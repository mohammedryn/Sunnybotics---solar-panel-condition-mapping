"""
GPS measurement simulation (RF-02 support, sets up RF-06).

Models the antenna-reported position as:
    true_camera_position -> (lever-arm offset removed) -> true_antenna_position
    true_antenna_position + per-mission bias + per-fix jitter -> reported position

The lever-arm offset (camera not co-located with the GPS antenna) is a
fixed, known/calibrated transform - it does not add uncertainty by
itself, it just means the algorithm consuming these fixes must know to
apply the same offset back out. Modeled explicitly so that assumption
is never implicit.
"""
import numpy as np

from . import config


def lever_arm_offset_world(heading_rad=config.ROW_HEADING_RAD):
    """Camera offset from GPS antenna, rotated from robot body frame
    (forward, right) into world (east, north) frame."""
    forward_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    right_vec = np.array([np.sin(heading_rad), -np.cos(heading_rad)])
    return config.LEVER_ARM_FORWARD_M * forward_vec + config.LEVER_ARM_RIGHT_M * right_vec


def sample_mission_bias(rng: np.random.Generator):
    """One bias vector (east, north) sampled once per mission - represents
    multipath/atmospheric drift that persists for the duration of a pass."""
    return rng.normal(0.0, config.GPS_MISSION_BIAS_STD_M, size=2)


def simulate_gps_fix(true_camera_east, true_camera_north, mission_bias, rng: np.random.Generator,
                      heading_rad=config.ROW_HEADING_RAD):
    """Returns the GPS-reported ANTENNA position (east, north) - what a
    real receiver would output. True camera position is never handed to
    the algorithm directly; that's the whole point of RF-06."""
    offset = lever_arm_offset_world(heading_rad)
    true_antenna_east = true_camera_east - offset[0]
    true_antenna_north = true_camera_north - offset[1]

    jitter = rng.normal(0.0, config.GPS_FIX_JITTER_STD_M, size=2)
    reported_east = true_antenna_east + mission_bias[0] + jitter[0]
    reported_north = true_antenna_north + mission_bias[1] + jitter[1]
    return reported_east, reported_north

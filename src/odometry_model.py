"""
Simulated wheel-encoder odometry (used by RF-06's association filter).

This is a second, independent along-track position sensor. Its noise
does not share the GPS's per-mission bias, which is exactly why fusing
the two later (in associate_panels.py) can catch the case where an
entire row pass is shifted by a constant GPS offset - odometry still
preserves the correct relative spacing between consecutive panels even
when GPS does not.

No long-range drift is modeled: a single row pass is ~20m, short enough
that unmodeled wheel-encoder drift is negligible. Documented
simplification - would need drift modeling for longer routes.
"""
import numpy as np

from . import config


def simulate_odometry_delta(true_delta_m: float, rng: np.random.Generator) -> float:
    """Reported along-track displacement since the previous capture."""
    return true_delta_m + rng.normal(0.0, config.ODOM_STEP_STD_M)

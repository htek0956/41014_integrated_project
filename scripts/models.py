"""Pure models: explicit inputs, no simulator access or mutable global state.

Adapt subject tutorials first and reuse suitable library functions.
Pose order is (x, y, theta), in world-frame metres/metres/radians.
Positive heading is counterclockwise. Use one angle-wrap convention.
"""

import numpy as np


def wrap(angle):
    """Wrap an angle or array of angles to [-pi, pi), in radians.

    Adapted from Assignment 3. In particular, +pi maps to -pi.
    """
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def wheel_increments_to_motion(delta_left, delta_right, geometry):
    """Convert wheel increments (rad) to (delta_s, delta_theta) in m/rad.

    geometry supplies wheel radius and separation, both in metres.
    """
    raise NotImplementedError("Adapt the differential-drive encoder model.")


def propagate_pose(pose, motion):
    """Predict pose from distance/heading increments, including straight motion."""
    raise NotImplementedError("Adapt the tutorial's discrete unicycle model.")


def motion_jacobians(pose, motion):
    """Return propagation derivatives with respect to pose and motion.

    Motion covariance must describe the increments used in this model.
    """
    raise NotImplementedError("Derive state and motion-noise Jacobians.")


def tracking_error_model(speed, curvature, dt):
    """Return Ad, Bd for cross-track/heading errors and yaw-rate correction.

    Units: speed in m/s, signed curvature in 1/m, controller interval in s.
    Match signs to tracking_errors. Alignment/stopping needs separate logic.
    """
    raise NotImplementedError("Linearise and discretise the tracking-error model.")

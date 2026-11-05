import numpy as np


def wrap(angle):

    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def wheel_increments_to_motion(delta_left, delta_right, geometry):

    raise NotImplementedError("Adapt the differential-drive encoder model.")


def propagate_pose(pose, motion):

    raise NotImplementedError("Adapt the tutorial's discrete unicycle model.")


def motion_jacobians(pose, motion):

    raise NotImplementedError("Derive state and motion-noise Jacobians.")


def tracking_error_model(speed, curvature, dt):

    raise NotImplementedError("Linearise and discretise the tracking-error model.")

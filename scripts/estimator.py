"""Encoder-camera EKF. Simulator truth is not an input."""


class PoseEKF:
    """Own pose, covariance, previous encoder samples and measurement times."""

    def __init__(self, initial_pose, initial_covariance, geometry):
        # Pose order: world-frame x (m), y (m), theta (rad).
        # Define process/measurement noise for the selected sensor models.
        raise NotImplementedError("Initialise the EKF and noise assumptions.")

    def predict(self, encoder_measurement):
        """Predict once from new encoder motion, using models.py.

        First sample establishes the encoder baseline. Subsequent increments
        cover the interval since the previous sample, not a command interval.
        """
        raise NotImplementedError("Adapt tutorial prediction to wheel encoders.")

    def correct(self, camera_observation):
        """Fuse one new valid camera observation at its measurement time.

        Choose camera-derived world pose or marker range/bearing.
        Define transforms, covariance and angular residuals consistently.
        """
        raise NotImplementedError("Adapt the tutorial measurement update.")

    def get_estimate(self):
        """Return pose, covariance and timestamp without sharing mutable state."""
        raise NotImplementedError("Return the current estimated state.")

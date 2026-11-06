"""Focused EKF checks with known answers, independent of MuJoCo."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from estimator import PoseEKF
from models import wrap


class PoseEKFTests(unittest.TestCase):
    def make_filter(self, pose=(0, 0, 0), covariance=None):
        if covariance is None:
            covariance = np.diag([0.04, 0.04, 0.01])
        return PoseEKF(pose, covariance,
                       {"wheel_radius": 0.033, "wheel_base": 0.288})

    def test_predictions_accumulate_without_recounting(self):
        ekf = self.make_filter()
        # Isolate motion propagation from the tuned process-noise covariance.
        ekf.Q[:] = 0
        initial_covariance = ekf.covariance.copy()
        for angle, distance in [(10, 0), (11, .033), (12, .066),
                                (12, .066), (11, .033)]:
            ekf.predict({"left_angle": angle, "right_angle": angle})
            np.testing.assert_allclose(ekf.state, [distance, 0, 0], atol=1e-12)
            F = np.eye(3)
            F[1, 2] = distance
            np.testing.assert_allclose(ekf.covariance,
                                       F @ initial_covariance @ F.T, atol=1e-12)

    def test_turn_direction(self):
        ekf = self.make_filter()
        ekf.predict({"left_angle": 10, "right_angle": 10})
        ekf.predict({"left_angle": 9, "right_angle": 11})
        np.testing.assert_allclose(ekf.state, [0, 0, .066 / .288], atol=1e-12)

    def test_missing_camera_leaves_prediction_unchanged(self):
        ekf = self.make_filter()
        state, covariance = ekf.state.copy(), ekf.covariance.copy()
        ekf.correct({"valid": False, "pose": None, "covariance": None})
        np.testing.assert_array_equal(ekf.state, state)
        np.testing.assert_array_equal(ekf.covariance, covariance)

    def test_equal_uncertainties_give_half_correction(self):
        ekf = self.make_filter()
        P = ekf.covariance.copy()
        ekf.correct({"valid": True, "pose": [0.1, -.2, .1], "covariance": P})
        np.testing.assert_allclose(ekf.state, [.05, -.1, .05], atol=1e-12)
        np.testing.assert_allclose(ekf.covariance, P / 2, atol=1e-12)

    def test_noisy_camera_has_less_influence(self):
        results = []
        for multiplier in [1, 100]:
            ekf = self.make_filter()
            ekf.correct({"valid": True, "pose": [1, 0, 0],
                         "covariance": multiplier * ekf.covariance.copy()})
            results.append(ekf.state[0])
        self.assertLess(results[1], results[0])

    def test_heading_crosses_wrap_boundary(self):
        ekf = self.make_filter(pose=(0, 0, np.deg2rad(179)))
        ekf.correct({"valid": True, "pose": [0, 0, np.deg2rad(-177)],
                     "covariance": ekf.covariance.copy()})
        self.assertGreaterEqual(ekf.state[2], -np.pi)
        self.assertLess(ekf.state[2], np.pi)
        self.assertAlmostEqual(float(wrap(ekf.state[2] - np.deg2rad(-179))), 0)

    def test_agreement_reduces_uncertainty_without_moving_pose(self):
        ekf = self.make_filter(pose=(2, 1, .3))
        state, P = ekf.state.copy(), ekf.covariance.copy()
        ekf.correct({"valid": True, "pose": state.copy(), "covariance": P.copy()})
        np.testing.assert_allclose(ekf.state, state, atol=1e-12)
        np.testing.assert_allclose(ekf.covariance, P / 2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

"""Known-answer checks for one-step error and sample covariance calculations."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_process_noise import prediction_errors, error_statistics, calibration_report
from models import propagate_pose, wheel_increments_to_motion, wrap


class ProcessNoiseCalibrationTests(unittest.TestCase):
    def make_log(self, residuals):
        settings = {"ekf_dt": .01, "wheel_radius": .033, "wheel_base": .288}
        truth = [np.array([0., 0., np.pi - .001])]
        angles = [np.zeros(2)]
        for residual in residuals:
            motion = wheel_increments_to_motion(.1, .2, settings)
            ending = np.asarray(propagate_pose(truth[-1], motion)) + residual
            ending[2] = wrap(ending[2])
            truth.append(ending)
            angles.append(angles[-1] + [.1, .2])
        return {"time": .01*np.arange(len(truth)), "truth": truth,
                "encoder_angles": angles, "settings": settings,
                "body_speed": np.tile([.1, .1], (len(truth), 1)),
                "phase": ["navigation"]*len(truth), "contact_steps": [0]*len(truth)}

    def test_exact_motion_has_zero_one_step_error(self):
        log = self.make_log(np.zeros((3, 3)))
        error, _ = prediction_errors(log)
        np.testing.assert_allclose(error, 0, atol=1e-12)

    def test_each_interval_starts_from_truth_and_wraps_heading(self):
        residuals = np.array([[.001, -.002, .003], [.002, .001, -.004], [-.001, .002, .005]])
        error, _ = prediction_errors(self.make_log(residuals))
        np.testing.assert_allclose(error, residuals, atol=1e-12)

    def test_bias_is_separate_from_sample_spread(self):
        residuals = np.array([[-.001, 0, -.01], [0, 0, 0], [.001, 0, .01]])
        bias = np.array([.002, .004, .02])
        stats = error_statistics(residuals + bias)
        np.testing.assert_allclose(stats["mean_error_m_m_rad"], bias, atol=1e-12)
        np.testing.assert_allclose(stats["prediction_error_sd_m_m_rad"], [.001, 0, .01], atol=1e-12)
        np.testing.assert_allclose(np.diag(stats["candidate_Q_diagonal"]), [1e-6, 0, 1e-4])
        self.assertAlmostEqual(stats["candidate_Q_full"][0][2], 1e-5)

    def test_different_prediction_interval_is_rejected(self):
        log = self.make_log(np.zeros((3, 3)))
        log["time"] *= 2
        with self.assertRaises(ValueError):
            prediction_errors(log)

    def test_report_handles_empty_motion_groups(self):
        log = self.make_log(np.zeros((3, 3)))
        error, _ = prediction_errors(log)
        report = calibration_report(log, error)
        self.assertEqual(report["groups"]["all"]["intervals"], 3)
        self.assertEqual(report["groups"]["turning"]["intervals"], 3)
        self.assertEqual(report["groups"]["stationary"]["intervals"], 0)


if __name__ == "__main__":
    unittest.main()

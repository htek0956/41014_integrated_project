"""Checks for encoder resolution, continuous travel and count-derived speed."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from models import encoder_angle_measurement


class EncoderModelTests(unittest.TestCase):
    def test_angles_are_counts_with_half_count_error_bound(self):
        step = 2*np.pi/4096
        angles = np.linspace(-20, 20, 10001)
        measured = encoder_angle_measurement(angles)
        np.testing.assert_allclose(measured/step, np.round(measured/step), atol=1e-10)
        self.assertLessEqual(np.max(np.abs(measured - angles)), step/2 + 1e-12)

    def test_preserves_revolutions_reverse_and_stationary_readings(self):
        step = 2*np.pi/4096
        angles = np.array([2*np.pi - .6*step, 2*np.pi + .6*step, -4*np.pi])
        measured = encoder_angle_measurement(angles)
        np.testing.assert_allclose(np.diff(measured)[:1], [2*step])
        self.assertAlmostEqual(measured[-1], -4*np.pi)
        np.testing.assert_array_equal(encoder_angle_measurement(angles), measured)

    def test_count_difference_changes_speed_even_with_small_angle_error(self):
        step = 2*np.pi/4096
        dt = .002
        angles = step * np.array([0, .4, .8, 1.2])
        measured = encoder_angle_measurement(angles)
        ideal_speed = np.diff(angles)/dt
        measured_speed = np.diff(measured)/dt
        np.testing.assert_allclose(measured_speed, [0, step/dt, 0])
        self.assertGreater(np.max(np.abs(measured_speed - ideal_speed)), 0)

    def test_invalid_resolution_or_angle_is_rejected(self):
        for counts in [0, -1, 1.5, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                encoder_angle_measurement(0, counts)
        with self.assertRaises(ValueError):
            encoder_angle_measurement([0, np.nan])


if __name__ == "__main__":
    unittest.main()

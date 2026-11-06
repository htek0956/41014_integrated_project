"""Check encoder history and PI timing at the actual robot boundary."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robot import Robot


class RobotEncoderTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = Path(__file__).resolve().parents[1] / "robotis_tb3/scene_turtlebot3_waffle_pi.xml"
        cls.robot = Robot(scene, max_yaw_rate=2, show_viewer=False,
                          physics_dt=.002, encoder_dt=.002, pi_dt=.01,
                          encoder_speed_window=.02)

    @classmethod
    def tearDownClass(cls):
        cls.robot.close()

    def setUp(self):
        self.robot.reset()

    def test_repeated_reads_do_not_change_history(self):
        robot = self.robot
        first = robot.read_encoders()
        for _ in range(5):
            self.assertEqual(robot.read_encoders(), first)
        self.assertEqual(len(robot.encoder_history), 1)
        first["left_angle"] = 999
        self.assertNotEqual(robot.read_encoders()["left_angle"], 999)

    def test_pi_updates_and_holds_torque(self):
        robot = self.robot
        controller = robot.wheel_controllers[0]
        updates = []
        with patch.object(controller, "update", wraps=controller.update) as update:
            for step in range(26):
                previous = robot.wheel_torques.copy()
                robot.set_body_command((.1, 0))
                robot.update_wheel_pi(robot.read_encoders())
                if robot.pi_updated:
                    updates.append(robot.time)
                else:
                    np.testing.assert_array_equal(robot.wheel_torques, previous)
                robot.step()
            self.assertEqual(update.call_count, 6)
            np.testing.assert_allclose([call.args[2] for call in update.call_args_list], .01)
        np.testing.assert_allclose(np.diff(updates), .01, atol=1e-12)

    def test_quantised_angles_and_rolling_speed(self):
        robot = self.robot
        for _ in range(50):
            robot.apply_body_command((.1, 0))
            robot.step()
        self.assertEqual(len(robot.encoder_history), 11)
        oldest, newest = robot.encoder_history[0], robot.encoder_history[-1]
        self.assertAlmostEqual(newest[0] - oldest[0], .02)
        reading = robot.read_encoders()
        expected = (np.asarray(newest[1:]) - oldest[1:]) / (newest[0] - oldest[0])
        np.testing.assert_allclose([reading["left_speed"], reading["right_speed"]], expected)
        angles = np.array([reading["left_angle"], reading["right_angle"]])
        counts = angles / (2*np.pi/4096)
        np.testing.assert_allclose(counts, np.round(counts), atol=1e-10)
        truth = robot.read_encoder_truth()
        self.assertLessEqual(np.max(np.abs(angles - [truth["left_angle"], truth["right_angle"]])),
                             np.pi/4096 + 1e-12)

    def test_reset_clears_history_and_restarts_pi(self):
        robot = self.robot
        for _ in range(15):
            robot.apply_body_command((.1, 0))
            robot.step()
        robot.reset()
        self.assertEqual(len(robot.encoder_history), 1)
        self.assertEqual(robot.read_encoders()["left_speed"], 0)
        self.assertEqual(robot.read_encoders()["right_speed"], 0)
        robot.apply_body_command((0, 0))
        self.assertTrue(robot.pi_updated)
        np.testing.assert_array_equal(robot.wheel_torques, 0)


if __name__ == "__main__":
    unittest.main()

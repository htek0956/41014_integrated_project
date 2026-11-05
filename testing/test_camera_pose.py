from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from robot import Robot


class CameraPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = Robot(ROOT / 'robotis_tb3/scene_loop.xml', max_yaw_rate=2, show_viewer=False)

    @classmethod
    def tearDownClass(cls):
        cls.robot.close()

    def test_station_pose_and_no_truth_dependency(self):
        robot = self.robot
        robot.reset((1.275, 0, -np.pi/2))
        frame = robot.read_camera()
        frame['timestamp'] = 12.34


        original = robot.get_truth
        robot.get_truth = lambda: (_ for _ in ()).throw(AssertionError('Truth leak'))
        try:
            first = robot.read_camera_pose(frame)
            robot.reset((6, 4, .4))
            second = robot.read_camera_pose(frame)
        finally:
            robot.get_truth = original
        self.assertTrue(first['valid'])
        self.assertEqual(first['timestamp'], 12.34)
        self.assertIn(0, first['ids_used'])
        np.testing.assert_allclose(first['pose'], second['pose'], atol=1e-10)
        self.assertLess(np.linalg.norm(first['pose'][:2]-[1.275, 0]), .02)
        self.assertLess(abs(first['pose'][2]+np.pi/2), .03)
        self.assertEqual(first['covariance'].shape, (3, 3))
        self.assertGreater(np.linalg.eigvalsh(first['covariance']).min(), 0)

    def test_marker_loss_never_returns_stale_pose(self):
        robot = self.robot
        robot.reset((1.275, 0, -np.pi/2))
        self.assertTrue(robot.read_camera_pose()['valid'])
        missing = robot.read_camera_pose(dict(timestamp=45., rgb=np.full((480, 640, 3), 127, np.uint8)))
        self.assertFalse(missing['valid'])
        self.assertIsNone(missing['pose'])
        self.assertIsNone(missing['covariance'])
        self.assertEqual(missing['ids_used'], [])
        self.assertEqual(missing['timestamp'], 45.)


if __name__ == '__main__':
    unittest.main()

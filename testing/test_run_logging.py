"""Check sample timing, existing caller data and selectable report exports."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import main
import evaluation
from evaluation import camera_innovation_nis
import plot_controller_results
import plot_estimator


class RunLoggingTests(unittest.TestCase):
    def test_all_reports_save_before_opening_windows(self):
        logger = evaluation.RunLogger.__new__(evaluation.RunLogger)
        logger.reports = ("estimator", "controller", "body", "pi")
        logger.estimator = {"time": [0.]}
        logger.control = {"time": [0.]}
        calls = Mock()
        with patch.object(plot_estimator, "save_estimator_results") as estimator_save, \
             patch.object(plot_controller_results, "save_controller_results") as control_save, \
             patch("matplotlib.pyplot.show") as show, \
             patch("matplotlib.pyplot.close"), patch.object(evaluation, "SHOW_REPORTS", True):
            calls.attach_mock(estimator_save, "estimator")
            calls.attach_mock(control_save, "controller")
            calls.attach_mock(show, "show")
            logger.save_reports(show=True)
        self.assertEqual([call[0] for call in calls.mock_calls],
                         ["estimator", "controller", "show"])
        self.assertFalse(estimator_save.call_args.kwargs["show"])
        self.assertFalse(control_save.call_args.kwargs["show"])
        self.assertFalse(estimator_save.call_args.kwargs["close"])

    def test_innovation_uses_wrapped_heading_without_changing_filter(self):
        estimator = SimpleNamespace(
            state=np.array([0., 0., np.deg2rad(179)]), covariance=np.eye(3))
        state = estimator.state.copy()
        measurement = dict(valid=True, pose=np.array([0., 0., np.deg2rad(-179)]),
                           covariance=np.eye(3))
        self.assertAlmostEqual(camera_innovation_nis(estimator, measurement),
                               np.deg2rad(2)**2 / 2)
        np.testing.assert_array_equal(estimator.state, state)
        self.assertTrue(np.isnan(camera_innovation_nis(estimator, {"valid": False})))

    def test_recording_and_report_selection_in_short_simulation(self):
        prediction = {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(main, "LOG_RESULTS", False), \
                 patch.object(evaluation, "FULL_SYSTEM_FOLDER", root / "full_system"), \
                 patch.object(plot_estimator, "RESULTS_FOLDER", root / "estimator"), \
                 patch.object(plot_controller_results, "RESULTS_FOLDER", root / "controller"):
                log = main.main(show_viewer=False, max_time=.42,
                                camera_diagnostics=True, prediction_log=prediction,
                                reports=("estimator", "controller", "body", "pi"))
                self.assertEqual(len(log["time"]), len(log["limited_command"]))
                np.testing.assert_allclose(np.diff(log["time"]), main.CONTROL_DT, atol=1e-10)
                np.testing.assert_allclose(np.diff(prediction["time"]), main.EKF_DT, atol=1e-10)
                self.assertEqual(len(log["wheel"]["time"]), len(log["wheel"]["target"]))
                self.assertEqual(sum(log["estimator"]["camera_updated"]), 3)
                self.assertTrue(log["camera_measurements"])
                self.assertIn("final_pose", log)
                self.assertIn("docked", log)
                self.assertIn("duration_s", prediction["settings"])
                shared = root / "full_system"
                self.assertEqual(len(list(shared.glob("*.png"))), 6)
                self.assertTrue((shared / "estimator_metrics_full_system.json").exists())
                self.assertTrue((shared / "controller_metrics_full_system.json").exists())
                self.assertTrue((shared / "estimator_data_full_system.npz").exists())
                self.assertTrue((shared / "run_data_full_system.npz").exists())
                self.assertFalse((root / "estimator").exists())
                self.assertFalse((root / "controller").exists())

                # Reusing the destination replaces matching files, not another run folder.
                plot_estimator.save_estimator_results(log["estimator"], output_folder=shared)
                plot_controller_results.save_controller_results(log, show=False, output_folder=shared)
                self.assertEqual(len(list(shared.iterdir())), 10)

                # A PI-only selection must not create controller/body figures.
                plot_controller_results.save_controller_results(log, show=False, reports=("pi",))
                images = list((root / "controller/run_001").glob("*.png"))
                self.assertEqual([p.name for p in images], ["pi_loop_001.png"])
                saved = plot_controller_results.load_controller_results(
                    root / "controller/run_001/run_data_001.npz")
                np.testing.assert_allclose(saved["pose"], log["pose"])

    def test_calibration_without_reports(self):
        prediction = {}
        with patch.object(main, "LOG_RESULTS", False):
            result = main.main(show_viewer=False, max_time=.04,
                               prediction_log=prediction,
                               reports=())
        self.assertIsNone(result)
        self.assertGreater(len(prediction["time"]), 1)
        self.assertEqual(prediction["settings"]["ekf_dt"], main.EKF_DT)

    def test_camera_dropout_and_recovery_are_recorded_at_image_times(self):
        with patch.object(main, "LOG_RESULTS", True), \
             patch.object(main, "CAMERA_DROPOUTS", True), \
             patch.object(main, "CAMERA_DROPOUT_PERIODS", ((.2, .4),)):
            log = main.main(show_viewer=False, max_time=.42, reports=())
        data = log["estimator"]
        updated = np.asarray(data["camera_updated"])
        np.testing.assert_allclose(np.asarray(data["time"])[updated], [0., .2, .4])
        np.testing.assert_array_equal(np.asarray(data["camera_valid"])[updated],
                                      [True, False, True])


if __name__ == "__main__":
    unittest.main()

"""Check estimator evaluation conventions and report round trips."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_estimator import (estimator_diagnostics, estimator_metrics,
                            load_estimator_results, save_estimator_results)


class EstimatorReportTests(unittest.TestCase):
    def make_log(self):
        return {
            "time": [0., 1.],
            "truth": [[0, 0, np.deg2rad(-179)], [.1, 0, 0]],
            "estimate": [[0, 0, np.deg2rad(179)], [0, 0, 0]],
            "covariance": np.tile(np.diag([.01, .01, .01]), (2, 1, 1)),
            "camera_updated": [True, True], "camera_valid": [True, False],
            "camera_pose": [[0, 0, np.deg2rad(-179)], [np.nan]*3],
            "tag_count": [2, 0], "nis": [3., np.nan],
            "reprojection_rms_px": [1., np.nan],
            "settings": {"process_covariance_per_prediction": np.zeros((3, 3)).tolist()},
        }

    def test_wrapped_errors_and_known_metrics(self):
        log = self.make_log()
        d = estimator_diagnostics(log)
        self.assertAlmostEqual(d["error"][0, 2], np.deg2rad(2))
        np.testing.assert_allclose(d["sigma"], .1)
        self.assertAlmostEqual(d["nees"][1], 1.)
        metrics = estimator_metrics(log, d)
        self.assertAlmostEqual(metrics["position_rmse_m"], .1/np.sqrt(2))
        self.assertEqual(metrics["camera_valid_fraction"], .5)

    def test_invalid_covariance_is_reported(self):
        log = self.make_log()
        log["covariance"][0, 0, 0] = -.01
        d = estimator_diagnostics(log)
        self.assertTrue(np.isnan(d["sigma"][0, 0]))
        self.assertTrue(np.isnan(d["nees"][0]))
        self.assertEqual(estimator_metrics(log, d)["invalid_covariance_samples"], 1)

    def test_save_reload_and_preserve_previous_runs(self):
        import matplotlib
        matplotlib.use("Agg")
        log = self.make_log()
        with tempfile.TemporaryDirectory() as root:
            first = save_estimator_results(log, results_folder=root)
            second = save_estimator_results(log, results_folder=root)
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(first.glob("*.png"))), 3)
            restored = load_estimator_results(next(first.glob("*.npz")))
            np.testing.assert_allclose(restored["estimate"], log["estimate"])
            self.assertEqual(estimator_metrics(restored), estimator_metrics(log))


if __name__ == "__main__":
    unittest.main()

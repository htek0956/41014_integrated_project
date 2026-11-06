"""Measure one-step encoder prediction errors to propose EKF process covariance.

Truth resets the predictor at the START of every interval, only in this offline
calibration. This measures new motion error, not accumulated EKF or camera error.
The operational filter and its Q are not changed by this script.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from models import propagate_pose, wheel_increments_to_motion, wrap


RESULTS_FOLDER = Path(__file__).resolve().parent / "figures" / "process_noise"


def prediction_errors(log):

    times = np.asarray(log["time"], dtype=float)
    truth = np.asarray(log["truth"], dtype=float)
    encoders = np.asarray(log["encoder_angles"], dtype=float)
    if times.ndim != 1 or len(times) < 3:
        raise ValueError("At least three timestamped samples are needed for covariance.")
    if truth.shape != (len(times), 3) or encoders.shape != (len(times), 2):
        raise ValueError("Expected one [x, y, heading] and [left, right] angle pair per time.")
    if not all(np.all(np.isfinite(a)) for a in (times, truth, encoders)):
        raise ValueError("Calibration samples must be finite.")

    intervals = np.diff(times)
    dt = log["settings"]["ekf_dt"]
    if not np.allclose(intervals, dt, atol=1e-9, rtol=0):
        raise ValueError("Recorded intervals must match the EKF prediction interval.")

    increments = np.diff(encoders, axis=0)
    predicted = []
    for starting_truth, wheel_change in zip(truth[:-1], increments):
        motion = wheel_increments_to_motion(*wheel_change, log["settings"])
        predicted.append(propagate_pose(starting_truth, motion))

    error = truth[1:] - np.asarray(predicted)
    error[:, 2] = wrap(error[:, 2])

    return error, np.asarray(predicted)


def error_statistics(errors):

    errors = np.asarray(errors, dtype=float)
    if len(errors) < 2:
        return {"intervals": len(errors), "note": "Too few intervals for sample covariance."}

    mean = errors.mean(axis=0)
    covariance = np.cov(errors, rowvar=False, ddof=1)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0))

    # Consecutive errors can be correlated (slip and count differencing).
    # Report that limitation rather than assuming a white-noise fit is exact.
    lag_one = []
    for column in range(3):
        previous, following = errors[:-1, column], errors[1:, column]
        if len(previous) < 2 or min(np.std(previous), np.std(following)) < 1e-15:
            lag_one.append(None)
        else:
            lag_one.append(float(np.corrcoef(previous, following)[0, 1]))

    return {
        "intervals": len(errors),
        "mean_error_m_m_rad": mean.tolist(),
        "mean_error_mm_mm_deg": (mean * [1000, 1000, 180/np.pi]).tolist(),
        "prediction_error_sd_m_m_rad": sd.tolist(),
        "prediction_error_sd_mm_mm_deg": (sd * [1000, 1000, 180/np.pi]).tolist(),
        "candidate_Q_full": covariance.tolist(),
        "candidate_Q_diagonal": np.diag(np.diag(covariance)).tolist(),
        # RMS includes systematic error; SD describes spread about the mean.
        "prediction_error_rmse_m_m_rad": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "lag_one_correlation_x_y_heading": lag_one,
    }


def calibration_report(log, error):

    # Use true endpoint speeds to distinguish actual stationary, straight and
    # turning motion, rather than treating commanded speed as physical motion.
    body_speed = np.asarray(log["body_speed"], dtype=float)
    forward = np.max(np.abs(np.column_stack((body_speed[:-1, 0], body_speed[1:, 0]))), axis=1)
    yaw = np.max(np.abs(np.column_stack((body_speed[:-1, 1], body_speed[1:, 1]))), axis=1)
    stationary = (forward < .01) & (yaw < .02)
    turning = ~stationary & (yaw >= .05)
    phase = np.asarray(log["phase"])[1:]
    contact = np.diff(np.asarray(log["contact_steps"])) > 0
    groups = {
        "all": np.ones(len(error), dtype=bool),
        "stationary": stationary,
        "straight_or_small_turn": ~stationary & ~turning,
        "turning": turning,
        "navigation": phase == "navigation",
        "docking": phase == "docking",
        "with_obstacle_contact": contact,
        "without_obstacle_contact": ~contact,
    }

    report = {
        "settings": log["settings"],
        "state_order": ["world_x_m", "world_y_m", "heading_rad"],
        "error_definition": "true ending pose minus prediction from true starting pose; heading wrapped",
        "covariance_units": "m^2, m*rad and rad^2 according to row/column; per EKF interval",
        "method": "Sample covariance about the measured mean, ddof=1; diagonal candidate discards correlations.",
        "limitations": [
            "A single deterministic simulation run is not a repeated-trial uncertainty calibration.",
            "Nonzero mean indicates bias; increasing Q does not repair the motion model.",
            "One-step covariance does not capture persistent error correlations or all long-term drift.",
            "World-frame error covariance depends on the route and its distribution of headings and speeds.",
            "Grouped lag-one correlations use adjacent selected rows, which can span gaps between conditions.",
            "Validate candidates on separate varied runs before applying them; EKF Q is not modified.",
        ],
        "classification": {"stationary_forward_threshold_m_s": .01,
                           "stationary_yaw_threshold_rad_s": .02,
                           "turning_yaw_threshold_rad_s": .05},
        "groups": {name: error_statistics(error[mask]) for name, mask in groups.items()},
    }
    return report


def plot_calibration(log, error, report):

    import matplotlib.pyplot as plt

    t = np.asarray(log["time"])[1:]
    stats = report["groups"]["all"]
    mean = np.asarray(stats["mean_error_m_m_rad"])
    sd = np.asarray(stats["prediction_error_sd_m_m_rad"])
    # A one-second rolling SD shows how variability changes around the route.
    # Each window uses at least two samples, with sample SD (ddof=1).
    window = max(2, round(1/log["settings"]["ekf_dt"]))
    rolling_sd = np.full_like(error, np.nan)
    for end in range(2, len(error) + 1):
        rolling_sd[end-1] = np.std(error[max(0, end-window):end], axis=0, ddof=1)

    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, layout="constrained")
    figure.suptitle("One-step motion prediction errors around the loop\n"
                   f"Quantised encoders; interval {1000*log['settings']['ekf_dt']:g} ms; "
                   "truth used only for offline calibration")
    for i, (name, scale, unit) in enumerate((("x", 1000, "mm"), ("y", 1000, "mm"),
                                           ("heading", 180/np.pi, "deg"))):
        axes[i, 0].plot(t, scale*error[:, i], lw=.7, label="One-step error")
        axes[i, 0].axhline(scale*mean[i], color="black", ls="--", label="Mean (bias)")
        axes[i, 0].axhspan(scale*(mean[i]-2*sd[i]), scale*(mean[i]+2*sd[i]),
                           alpha=.15, color="tab:blue", label="Mean +/-2 sample SD (not EKF bounds)")
        axes[i, 0].set_ylabel(f"{name} error [{unit}]")
        axes[i, 1].plot(t, scale*rolling_sd[:, i], lw=1, label="Rolling sample SD (up to 1 s)")
        axes[i, 1].axhline(scale*sd[i], color="black", ls="--", label="Whole-run sample SD")
        axes[i, 1].set_ylabel(f"{name} SD [{unit}]")
    for ax in axes.flat:
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Simulation time [s]")
    axes[-1, 1].set_xlabel("Simulation time [s]")
    return figure


def save_calibration(log, show=True, results_folder=RESULTS_FOLDER):

    import matplotlib.pyplot as plt

    error, predicted = prediction_errors(log)
    report = calibration_report(log, error)
    root = Path(results_folder)
    root.mkdir(parents=True, exist_ok=True)
    index = 1 + max((int(p.name[4:]) for p in root.glob("run_*")
                     if p.is_dir() and p.name[4:].isdigit()), default=0)
    while True:
        folder = root / f"run_{index:03d}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            index += 1

    arrays = {key: np.asarray(value) for key, value in log.items() if key != "settings"}
    arrays.update(settings=np.array(json.dumps(log["settings"])),
                  prediction_error=error, predicted_ending_pose=predicted)
    np.savez_compressed(folder / f"prediction_data_{index:03d}.npz", **arrays)
    (folder / f"calibration_{index:03d}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    figure = plot_calibration(log, error, report)
    figure.savefig(folder / f"prediction_variability_{index:03d}.png", dpi=150)

    stats = report["groups"]["all"]
    print(f"Prediction interval: {log['settings']['ekf_dt']:.4f} s; {stats['intervals']} intervals")
    print("Mean error [mm, mm, deg]:", np.asarray(stats["mean_error_mm_mm_deg"]))
    print("Error SD   [mm, mm, deg]:", np.asarray(stats["prediction_error_sd_mm_mm_deg"]))
    print("Candidate Q diagonal [m^2, m^2, rad^2]:")
    print(np.asarray(stats["candidate_Q_diagonal"]))
    print("Candidate only: check bias, correlated errors and separate validation runs before use.")
    print(f"Saved calibration: {folder}")
    if show:
        plt.show()
    plt.close(figure)
    return folder


def load_prediction_log(path):

    with np.load(path, allow_pickle=False) as saved:
        log = {key: saved[key].copy() for key in (
            "time", "truth", "encoder_angles", "ideal_encoder_angles",
            "body_speed", "phase", "contact_steps",
        )}
        log["settings"] = json.loads(str(saved["settings"]))
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Reanalyse a saved prediction_data_NNN.npz without driving.")
    parser.add_argument("--max-time", type=float, help="Limit simulated seconds; omit for the full loop and docking.")
    parser.add_argument("--viewer", action="store_true", help="Show the simulation while recording.")
    parser.add_argument("--no-show", action="store_true", help="Save results without opening the plot.")
    args = parser.parse_args()

    if args.input:
        log = load_prediction_log(args.input)
    else:
        import main

        # Reuse the real controller, PI, encoder model and scheduling. No
        # alternative calibration controller or EKF truth feedback is added.
        log = {}
        main.main(show_viewer=args.viewer, max_time=args.max_time,
                  estimator_diagnostics=False, prediction_log=log)

    save_calibration(log, show=not args.no_show)

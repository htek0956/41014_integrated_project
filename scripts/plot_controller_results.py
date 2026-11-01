"""Combined controller plots and indexed results.

Run without arguments for a new headless experiment, or use --input with a saved
run_data_NNN.npz to replot it. Every save creates a new run folder.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "figures" / "controller_tracking"


def controller_metrics(log):
    """Calculate full-run metrics, including startup and stopping."""
    cross_track = np.asarray(log["cross_track_error"])
    heading = np.rad2deg(log["heading_error"])
    wheel = {key: np.asarray(value) for key, value in log["wheel"].items()}
    clipped = np.abs(wheel["raw_torque"] - wheel["torque"]) > 1e-9
    command_limited = np.any(
        np.abs(wheel["body_command"] - wheel["limited_command"]) > 1e-9, axis=1
    )

    return {
        "completed": bool(log["completed"]),
        "duration_s": float(log["duration"]),
        "endpoint_error_m": float(log["endpoint_error"]),
        "cross_track_rms_m": float(np.sqrt(np.mean(cross_track**2))),
        "cross_track_max_abs_m": float(np.max(np.abs(cross_track))),
        "heading_rms_deg": float(np.sqrt(np.mean(heading**2))),
        "heading_max_abs_deg": float(np.max(np.abs(heading))),
        "wheel_speed_rms_error_rad_s": np.sqrt(np.mean(
            (wheel["target"] - wheel["measured"])**2, axis=0
        )).tolist(),
        "torque_rms_Nm": np.sqrt(np.mean(wheel["torque"]**2, axis=0)).tolist(),
        "torque_clipped_percent": (100 * np.mean(clipped, axis=0)).tolist(),
        "body_command_limited_percent": float(100 * np.mean(command_limited)),
    }


def plot_controller_results(log, metrics):
    """Build one figure. Ground-truth trajectory is compared to the desired track."""
    times = np.asarray(log["time"])
    poses = np.asarray(log["pose"])
    track = np.asarray(log["reference_path"])
    wheel = {key: np.asarray(value) for key, value in log["wheel"].items()}
    wheel_times = wheel["time"]
    settings = log["settings"]

    figure, axes = plt.subplots(5, 2, figsize=(15, 18), layout="constrained")
    figure.suptitle(
        "Paperclip controller diagnostic - ground-truth feedback\n"
        f"Reference speed {settings['reference_speed']:g} m/s | "
        f"LQR dt {settings['control_dt']:g} s | "
        f"K = {np.array2string(np.asarray(settings['lqr_gain']), precision=3)}",
        fontsize=14,
    )

    axis = axes[0, 0]
    axis.plot(track[:, 0], track[:, 1], "--", color="black", label="Desired track")
    axis.plot(poses[:, 0], poses[:, 1], color="tab:blue", label="Ground-truth motion")
    axis.scatter(*track[0], color="green", label="Start", zorder=5)
    axis.scatter(*track[-1], color="red", marker="x", label="Desired stop", zorder=5)
    axis.scatter(*log["final_pose"][:2], color="tab:orange", marker="+",
                 label="Final actual position", zorder=6)
    axis.set(title="Desired track and actual trajectory", xlabel="World x [m]",
             ylabel="World y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=8)

    axes[0, 1].axis("off")
    pi = settings["pi_gains"]
    text = (
        f"Completed: {metrics['completed']}\n"
        f"Simulation duration: {metrics['duration_s']:.2f} s\n"
        f"Final endpoint error: {metrics['endpoint_error_m']:.4f} m\n\n"
        f"Cross-track RMS: {metrics['cross_track_rms_m']:.4f} m\n"
        f"Cross-track max: {metrics['cross_track_max_abs_m']:.4f} m\n"
        f"Heading RMS: {metrics['heading_rms_deg']:.3f} deg\n"
        f"Heading max: {metrics['heading_max_abs_deg']:.3f} deg\n\n"
        f"Torque RMS L/R: {metrics['torque_rms_Nm'][0]:.4f} / "
        f"{metrics['torque_rms_Nm'][1]:.4f} N m\n"
        f"Torque clipped L/R: {metrics['torque_clipped_percent'][0]:.2f} / "
        f"{metrics['torque_clipped_percent'][1]:.2f} %\n"
        f"Body command limited: {metrics['body_command_limited_percent']:.2f} %\n\n"
        f"Left PI: Kp={pi[0]['kp']:g}, Ki={pi[0]['ki']:g}\n"
        f"Right PI: Kp={pi[1]['kp']:g}, Ki={pi[1]['ki']:g}\n"
        "Metrics include startup and stopping.\n"
        "Local cross-track error can omit along-path endpoint error.\n"
        "Clipping statistics use every physics-step sample."
    )
    axes[0, 1].text(0.03, 0.98, text, va="top", fontsize=10, linespacing=1.5)

    axes[1, 0].plot(times, log["cross_track_error"])
    axes[1, 0].set(title="Signed cross-track error", ylabel="Error [m]")
    axes[1, 1].plot(times, np.rad2deg(log["heading_error"]))
    axes[1, 1].set(title="Wrapped heading error", ylabel="Error [deg]")

    for column, (title, unit) in enumerate((
        ("Forward body speed", "Speed [m/s]"),
        ("Body yaw rate", "Yaw rate [rad/s]"),
    )):
        axis = axes[2, column]
        axis.plot(wheel_times, wheel["body_command"][:, column], "--",
                  label="Requested")
        axis.plot(wheel_times, wheel["limited_command"][:, column], ":",
                  label="After command limits")
        axis.plot(wheel_times, wheel["body_measured"][:, column],
                  label="Ground-truth measured", linewidth=1)
        axis.set(title=title, ylabel=unit)
        axis.legend(fontsize=8)

    for column, name in enumerate(("Left", "Right")):
        axis = axes[3, column]
        axis.plot(wheel_times, wheel["target"][:, column], "--", label="Target")
        axis.plot(wheel_times, wheel["measured"][:, column], label="Encoder")
        axis.set(title=f"{name} wheel speed", ylabel="Speed [rad/s]")
        axis.legend(fontsize=8)

        axis = axes[4, column]
        axis.plot(wheel_times, wheel["raw_torque"][:, column], "--",
                  label="PI request", linewidth=1)
        axis.plot(wheel_times, wheel["torque"][:, column],
                  label="Applied actuator command", linewidth=1)
        lower, upper = settings["torque_limits"][column]
        axis.axhline(lower, color="red", linestyle=":", label="Torque limits")
        axis.axhline(upper, color="red", linestyle=":")
        # Show applied effort clearly without losing the large unclipped PI
        # transients. The inset retains the full request range over the run.
        margin = 0.15 * (upper - lower)
        axis.set_ylim(lower - margin, upper + margin)
        inset = axis.inset_axes([0.63, 0.57, 0.34, 0.34])
        inset.plot(wheel_times, wheel["raw_torque"][:, column], linewidth=0.6)
        inset.set_title("Full PI request range [N m]", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(True, alpha=0.2)
        axis.set(title=f"{name} wheel actuator effort", ylabel="Torque [N m]")
        axis.legend(fontsize=8)

    for axis in axes.flat:
        if axis.axison:
            axis.grid(True, alpha=0.3)

    for axis in axes[1:].flat:
        axis.set_xlabel("Simulation time [s]")
        axis.axhline(0, color="grey", linewidth=0.5)
        if log["stop_time"] is not None:
            axis.axvline(log["stop_time"], color="grey", linestyle=":", linewidth=1)
    # Shared time extent helps relate tracking transients to motor behaviour.
    for axis in axes[1:].flat:
        axis.set_xlim(0, log["duration"])

    return figure


def save_controller_results(log, show=True):
    """Save image, numeric data and metrics without overwriting previous runs."""
    if not log["time"] or not log["wheel"]["time"]:
        raise ValueError("There are no controller samples to plot.")

    metrics = controller_metrics(log)
    figure = plot_controller_results(log, metrics)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    indices = [
        int(folder.name[4:])
        for folder in RESULTS_FOLDER.glob("run_*")
        if folder.is_dir() and folder.name[4:].isdigit()
    ]
    index = max(indices, default=0) + 1

    # Exclusive directory creation reserves a new index even if two runs save
    # simultaneously. Existing folders and images are never replaced.
    while True:
        folder = RESULTS_FOLDER / f"run_{index:03d}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            index += 1

    image_path = folder / f"controller_overview_{index:03d}.png"
    with image_path.open("xb") as output:
        figure.savefig(output, format="png", dpi=150)

    # Plain numeric arrays plus JSON metadata: no pickle required when loading.
    arrays = {
        key: np.asarray(log[key]) for key in (
            "time", "pose", "cross_track_error", "heading_error",
            "command", "limited_command", "reference_path",
        )
    }
    arrays.update({
        f"wheel_{key}": np.asarray(value)
        for key, value in log["wheel"].items()
    })
    metadata = {
        key: log[key] for key in (
            "settings", "completed", "duration", "endpoint_error",
            "stop_time", "final_pose",
        )
    }
    arrays["metadata"] = np.array(json.dumps(metadata))
    with (folder / f"run_data_{index:03d}.npz").open("xb") as output:
        np.savez_compressed(output, **arrays)

    with (folder / f"metrics_{index:03d}.json").open("x", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)

    print(f"Saved controller results: {folder}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return image_path


def load_controller_results(path):
    """Read a locally saved numeric run archive for plotting again."""
    with np.load(path, allow_pickle=False) as saved:
        log = json.loads(str(saved["metadata"]))
        log["wheel"] = {}
        for name in saved.files:
            if name.startswith("wheel_"):
                log["wheel"][name[6:]] = saved[name].copy()
            elif name != "metadata":
                log[name] = saved[name].copy()

    # Save/plot uses list emptiness checks for the outer sample lists.
    log["time"] = log["time"].tolist()
    log["wheel"]["time"] = log["wheel"]["time"].tolist()
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Saved run_data_NNN.npz file.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot.")
    args = parser.parse_args()

    if args.input is None:
        from main import main
        log = main(show_viewer=False)
    else:
        log = load_controller_results(args.input)

    save_controller_results(log, show=not args.no_show)

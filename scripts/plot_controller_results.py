import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_pi_results import plot_pi_results


RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "figures" / "controller_tracking"


def controller_metrics(log):

    cross_track = np.asarray(log["cross_track_error"])
    heading = np.rad2deg(log["heading_error"])
    wheel = {key: np.asarray(value) for key, value in log["wheel"].items()}
    clipped = np.abs(wheel["raw_torque"] - wheel["torque"]) > 1e-9
    command_limited = np.any(
        np.abs(wheel["body_command"] - wheel["limited_command"]) > 1e-9, axis=1
    )

    metrics = {
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
    metrics.update({key: log["settings"][key] for key in
                    ("lap_completed", "docked", "unwanted_contact_steps",
                     "station_contact_steps", "contact_geoms") if key in log["settings"]})
    return metrics


def plot_controller_results(log, metrics):

    times = np.asarray(log["time"])
    poses = np.asarray(log["pose"])
    track = np.asarray(log["reference_path"])
    settings = log["settings"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    figure.suptitle(
        "Path controller - feedback tracking\n"
        f"Reference speed {settings['reference_speed']:g} m/s | "
        f"Control dt {settings['control_dt']:g} s | "
        f"K = {np.array2string(np.asarray(settings['lqr_gain']), precision=3)}"
    )

    axis = axes[0, 0]
    axis.plot(track[:, 0], track[:, 1], "--", color="black", label="Desired track")
    axis.plot(poses[:, 0], poses[:, 1], label="Feedback pose")
    axis.scatter(*track[0], color="green", label="Start", zorder=5)
    axis.scatter(*track[-1], color="red", marker="x", label="Desired stop", zorder=5)
    axis.scatter(*log["final_pose"][:2], color="tab:orange", marker="+",
                 label="Final position", zorder=6)
    axis.set(title="Original route and feedback trajectory", xlabel="World x [m]",
             ylabel="World y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=8)

    axes[0, 1].axis("off")
    text = (
        f"Completed: {metrics['completed']}\n"
        f"Simulation duration: {metrics['duration_s']:.2f} s\n"
        f"Final endpoint error: {metrics['endpoint_error_m']:.4f} m\n\n"
        f"Cross-track RMS: {metrics['cross_track_rms_m']:.4f} m\n"
        f"Cross-track max: {metrics['cross_track_max_abs_m']:.4f} m\n"
        f"Heading RMS: {metrics['heading_rms_deg']:.3f} deg\n"
        f"Heading max: {metrics['heading_max_abs_deg']:.3f} deg\n"
        f"Body command limited: {metrics['body_command_limited_percent']:.2f} %\n\n"
        "Metrics include startup, stopping and docking.\n"
        "Cross-track error excludes along-path stopping error.\n"
        "Dotted vertical line: endpoint stop requested."
    )
    axes[0, 1].text(0.03, 0.95, text, va="top", fontsize=11, linespacing=1.6)

    axes[1, 0].plot(times, log["cross_track_error"])
    axes[1, 0].set(title="Signed cross-track error", ylabel="Error [m]")
    axes[1, 1].plot(times, np.rad2deg(log["heading_error"]))
    axes[1, 1].set(title="Wrapped heading error", ylabel="Error [deg]")

    for axis in axes[1]:
        axis.set_xlabel("Simulation time [s]")
        axis.set_xlim(0, log["duration"])
        axis.axhline(0, color="grey", linewidth=0.5)
        if log["stop_time"] is not None:
            axis.axvline(log["stop_time"], color="grey", linestyle=":")

    for axis in (axes[0, 0], *axes[1]):
        axis.grid(True, alpha=0.3)

    return figure


def plot_body_response(log):

    wheel = {key: np.asarray(value) for key, value in log["wheel"].items()}
    times = wheel["time"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                layout="constrained")
    figure.suptitle("Body response - requested motion versus ground truth")

    for column, (axis, title, unit) in enumerate(zip(
        axes, ("Forward body speed", "Body yaw rate"),
        ("Speed [m/s]", "Yaw rate [rad/s]"),
    )):
        axis.plot(times, wheel["body_command"][:, column], "--",
                  color="black", label="Requested")
        axis.plot(times, wheel["limited_command"][:, column], ":",
                  color="tab:orange", label="After command limits")
        axis.plot(times, wheel["body_measured"][:, column],
                  color="tab:blue", label="Ground-truth measured", linewidth=1)
        axis.set(title=title, ylabel=unit, xlim=(0, log["duration"]))
        axis.grid(True, alpha=0.3)
        axis.legend()
        if log["stop_time"] is not None:
            axis.axvline(log["stop_time"], color="grey", linestyle=":")

    axes[-1].set_xlabel("Simulation time [s]")
    return figure


def save_controller_results(log, show=True, reports=("controller", "body", "pi"),
                            output_folder=None, close=True):

    if not log["time"] or not log["wheel"]["time"]:
        raise ValueError("There are no controller samples to plot.")

    metrics = controller_metrics(log)
    unknown = set(reports) - {"controller", "body", "pi"}
    if unknown or not reports:
        raise ValueError("Choose at least one report from controller, body and pi.")
    figures = []
    if "controller" in reports:
        figures.append((plot_controller_results(log, metrics), "controller_overview"))
    if "body" in reports:
        figures.append((plot_body_response(log), "body_response"))
    if "pi" in reports:
        figures.append((plot_pi_results(log, metrics), "pi_loop"))

    if output_folder is not None:
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "full_system"
    else:
        RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
        index = 1 + max((int(p.name[4:]) for p in RESULTS_FOLDER.glob("run_*")
                         if p.is_dir() and p.name[4:].isdigit()), default=0)
        while True:
            folder = RESULTS_FOLDER / f"run_{index:03d}"
            try:
                folder.mkdir()
                break
            except FileExistsError:
                index += 1
        suffix = f"{index:03d}"

    image_path = folder / f"{figures[0][1]}_{suffix}.png"
    for report, name in figures:
        with (folder / f"{name}_{suffix}.png").open("wb") as output:
            report.savefig(output, format="png", dpi=150)


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
    metadata["metrics"] = metrics
    arrays["metadata"] = np.array(json.dumps(metadata))
    with (folder / f"run_data_{suffix}.npz").open("wb") as output:
        np.savez_compressed(output, **arrays)

    metrics_name = "controller_metrics" if output_folder is not None else "metrics"
    with (folder / f"{metrics_name}_{suffix}.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)

    print(f"Saved controller results: {folder}")

    if show:
        plt.show()
    elif close:
        for figure, _ in figures:
            plt.close(figure)

    return image_path


def load_controller_results(path):

    with np.load(path, allow_pickle=False) as saved:
        log = json.loads(str(saved["metadata"]))
        log["wheel"] = {}
        for name in saved.files:
            if name.startswith("wheel_"):
                log["wheel"][name[6:]] = saved[name].copy()
            elif name != "metadata":
                log[name] = saved[name].copy()


    log["time"] = log["time"].tolist()
    log["wheel"]["time"] = log["wheel"]["time"].tolist()
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Saved run_data_NNN.npz file.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot.")
    args = parser.parse_args()

    if args.input is None:
        import main as simulation
        simulation.LOG_RESULTS = True
        # This script saves its own reports below.
        log = simulation.main(show_viewer=False, reports=())
    else:
        log = load_controller_results(args.input)

    if log is not None:
        save_controller_results(log, show=not args.no_show)

"""Compare ideal and quantised encoder feedback with the same wheel PI.

This is a diagnostic, not a change to the robot's operational sensor interface.
Both trials estimate speed over the same rolling window and update PI at
the same slower control interval. Physics continues between PI updates.
This isolates quantisation, but does not emulate the motor's velocity register.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from models import encoder_angle_measurement, propagate_pose, wheel_increments_to_motion, wrap
from plot_pi_results import plot_pi_results
from robot import Robot


COUNTS_PER_REVOLUTION = 4096
TEST_DURATION = 11.0
SPEED_WINDOW = 0.02
PI_DT = 0.01
RESULTS_FOLDER = Path(__file__).resolve().parent / "figures" / "encoder_noise"


def wheel_targets(sim_time):

    # Rest, smoothly accelerate, hold, turn, reverse and stop.
    # Targets are wheel angular speeds [left, right], in rad/s.
    if sim_time < 1.0:
        return np.array([0.0, 0.0])

    if sim_time < 3.0:
        progress = (sim_time - 1.0) / 2.0
        speed = 4.0 * (3 * progress**2 - 2 * progress**3)
        return np.array([speed, speed])

    if sim_time < 5.0:
        return np.array([4.0, 4.0])

    if sim_time < 7.0:
        return np.array([3.0, 6.0])

    if sim_time < 9.0:
        return np.array([-3.0, -3.0])

    return np.array([0.0, 0.0])


def run_encoder_test(quantised_feedback, counts_per_revolution=COUNTS_PER_REVOLUTION,
                     show_viewer=False, speed_window=SPEED_WINDOW, control_dt=PI_DT):

    if not np.all(np.isfinite([speed_window, control_dt])) or min(speed_window, control_dt) <= 0:
        raise ValueError("Speed window and PI interval must be finite and positive.")
    encoder_angle_measurement(0.0, counts_per_revolution)

    scene = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
    robot = Robot(scene, max_yaw_rate=2.0, show_viewer=show_viewer)

    window_steps = int(round(speed_window / robot.physics_dt))
    control_steps = int(round(control_dt / robot.physics_dt))
    if (min(window_steps, control_steps) < 1
            or not np.isclose(window_steps * robot.physics_dt, speed_window, rtol=0, atol=1e-10)
            or not np.isclose(control_steps * robot.physics_dt, control_dt, rtol=0, atol=1e-10)):
        robot.close()
        raise ValueError("Speed window and PI interval must be whole multiples of the physics timestep.")

    log = {name: [] for name in (
        "time", "angle_truth", "angle_quantised", "speed_truth_interval",
        "speed_truth_instant", "speed_quantised", "truth_pose", "truth_body_speed",
        "odometry_ideal", "odometry_quantised",
        "control_updated", "speed_window_elapsed", "feedback_truth_interval",
    )}
    log["wheel"] = {name: [] for name in (
        "time", "target", "measured", "torque", "raw_torque",
    )}
    log["settings"] = {
        "feedback": "quantised" if quantised_feedback else "ideal",
        "counts_per_revolution": counts_per_revolution,
        "physics_dt": robot.physics_dt,
        "speed_estimator": "rolling angle difference / elapsed window, in both trials",
        "speed_window": speed_window,
        "control_dt": control_dt,
        "pi_gains": [{"kp": c.kp, "ki": c.ki} for c in robot.wheel_controllers],
        "torque_limits": robot.model.actuator_ctrlrange[robot.actuators].tolist(),
        "wheel_radius": robot.wheel_radius,
        "wheel_base": robot.wheel_separation,
    }
    geometry = log["settings"]

    try:
        initial = robot.read_encoder_truth()
        previous_angles = np.array([initial["left_angle"], initial["right_angle"]])
        previous_quantised = encoder_angle_measurement(previous_angles, counts_per_revolution)
        previous_time = robot.time
        history = deque(maxlen=window_steps + 1)

        targets = np.zeros(2)
        measured_speed = np.zeros(2)
        feedback_truth = np.zeros(2)
        torques = np.zeros(2)
        step = 0

        # Truth seeds odometry only for this labelled diagnostic. Subsequent
        # odometry uses wheel increments only, so slip remains visible.
        odometry_ideal = np.asarray(robot.get_truth())
        odometry_quantised = odometry_ideal.copy()
        wall_start = time.perf_counter()
        next_viewer_time = 0.0

        while robot.running() and robot.time < TEST_DURATION - 1e-10:
            encoders = robot.read_encoder_truth()
            angles = np.array([encoders["left_angle"], encoders["right_angle"]])
            quantised = encoder_angle_measurement(angles, counts_per_revolution)
            interval = robot.time - previous_time

            # First reading establishes a baseline; do not invent a speed from
            # an unknown previous position. The robot starts at rest.
            if interval > 0:
                ideal_increments = angles - previous_angles
                quantised_increments = quantised - previous_quantised

                odometry_ideal = np.asarray(propagate_pose(
                    odometry_ideal, wheel_increments_to_motion(*ideal_increments, geometry)))
                odometry_quantised = np.asarray(propagate_pose(
                    odometry_quantised, wheel_increments_to_motion(*quantised_increments, geometry)))
            # Read angles at the physics rate, but estimate speed over a longer
            # rolling window. This reduces count steps at the cost of delay.
            history.append((robot.time, angles.copy(), quantised.copy()))
            oldest_time, oldest_angles, oldest_quantised = history[0]
            elapsed = robot.time - oldest_time
            if elapsed > 0:
                ideal_speed = (angles - oldest_angles) / elapsed
                quantised_speed = (quantised - oldest_quantised) / elapsed
            else:
                ideal_speed = np.zeros(2)
                quantised_speed = np.zeros(2)

            control_updated = step % control_steps == 0
            if control_updated:
                measured_speed = (quantised_speed if quantised_feedback else ideal_speed).copy()
                feedback_truth = ideal_speed.copy()
                targets = wheel_targets(robot.time)

                # PI runs at control_dt, not physics_dt. Its integrator must use
                # this longer interval too. Torque is held until the next update.
                torques = np.array([
                    controller.update(target, measured, control_dt)
                    for controller, target, measured in zip(
                        robot.wheel_controllers, targets, measured_speed)
                ])
            robot.data.ctrl[robot.actuators] = torques

            log["time"].append(robot.time)
            log["angle_truth"].append(angles.copy())
            log["angle_quantised"].append(quantised.copy())
            log["speed_truth_interval"].append(ideal_speed.copy())
            log["speed_quantised"].append(quantised_speed.copy())
            log["speed_truth_instant"].append([encoders["left_speed"], encoders["right_speed"]])
            log["truth_pose"].append(robot.get_truth())
            log["truth_body_speed"].append(robot.get_truth_velocity())
            log["odometry_ideal"].append(odometry_ideal.copy())
            log["odometry_quantised"].append(odometry_quantised.copy())
            log["control_updated"].append(control_updated)
            log["speed_window_elapsed"].append(elapsed)
            log["feedback_truth_interval"].append(feedback_truth.copy())

            wheel = log["wheel"]
            wheel["time"].append(robot.time)
            wheel["target"].append(targets)
            wheel["measured"].append(measured_speed.copy())
            wheel["torque"].append(torques)
            wheel["raw_torque"].append([c.raw_torque for c in robot.wheel_controllers])

            previous_angles = angles.copy()
            previous_quantised = quantised.copy()
            previous_time = robot.time
            robot.step()
            step += 1

            if show_viewer and robot.time >= next_viewer_time:
                robot.sync_viewer()
                time.sleep(max(0, wall_start + robot.time - time.perf_counter()))
                next_viewer_time = robot.time + 1/30

        log["duration"] = robot.time
        log["stop_time"] = 9.0 if robot.time >= 9.0 else None
        return log

    finally:
        robot.close()


def encoder_metrics(log):

    wheel = {name: np.asarray(value) for name, value in log["wheel"].items()}
    angle_error = np.asarray(log["angle_quantised"]) - log["angle_truth"]
    speed_error = np.asarray(log["speed_quantised"]) - log["speed_truth_interval"]
    clipped = np.abs(wheel["raw_torque"] - wheel["torque"]) > 1e-9

    metrics = {
        # These keys are the existing plot_pi_results interface.
        "wheel_speed_rms_error_rad_s": np.sqrt(np.mean(
            (wheel["target"] - wheel["measured"])**2, axis=0)).tolist(),
        "torque_clipped_percent": (100 * clipped.mean(axis=0)).tolist(),
        "torque_rms_Nm": np.sqrt(np.mean(wheel["torque"]**2, axis=0)).tolist(),
        "true_wheel_tracking_rmse_rad_s": np.sqrt(np.mean(
            (wheel["target"] - log["speed_truth_instant"])**2, axis=0)).tolist(),
        "angle_error_max_abs_rad": np.max(np.abs(angle_error), axis=0).tolist(),
        "angle_error_sd_rad": np.std(angle_error, axis=0).tolist(),
        "speed_measurement_error_rmse_rad_s": np.sqrt(np.mean(speed_error**2, axis=0)).tolist(),
        "feedback_speed_measurement_error_rmse_rad_s": np.sqrt(np.mean(
            (wheel["measured"] - log["feedback_truth_interval"])**2, axis=0)).tolist(),
        "torque_change_rms_Nm": np.sqrt(np.mean(
            np.diff(wheel["torque"][np.asarray(log["control_updated"], dtype=bool)], axis=0)**2,
            axis=0)).tolist(),
        "pi_updates": int(np.count_nonzero(log["control_updated"])),
        "angle_per_count_rad": float(2*np.pi/log["settings"]["counts_per_revolution"]),
        "velocity_step_rad_s": float(2*np.pi/log["settings"]["counts_per_revolution"]
                                     / log["settings"]["speed_window"]),
    }

    for name in ("ideal", "quantised"):
        error = np.asarray(log[f"odometry_{name}"]) - log["truth_pose"]
        error[:, 2] = wrap(error[:, 2])
        metrics[f"odometry_{name}_position_rmse_m"] = float(np.sqrt(np.mean(
            np.sum(error[:, :2]**2, axis=1))))
        metrics[f"odometry_{name}_heading_rmse_deg"] = float(np.rad2deg(
            np.sqrt(np.mean(error[:, 2]**2))))

    return metrics


def plot_encoder_readings(log):

    t = np.asarray(log["time"])
    angle = np.asarray(log["angle_truth"])
    measured = np.asarray(log["angle_quantised"])
    speed = np.asarray(log["speed_truth_interval"])
    measured_speed = np.asarray(log["speed_quantised"])
    figure, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="row", layout="constrained")
    figure.suptitle(f"Encoder readings vs joint truth — {log['settings']['feedback']} PI feedback\n"
                   f"Speed window {1000*log['settings']['speed_window']:g} ms; "
                   f"PI interval {1000*log['settings']['control_dt']:g} ms; no random jitter")

    # A ramp zoom makes individual count steps visible instead of hiding them
    # beneath a multi-revolution angle plot.
    zoom = (t >= 1.8) & (t <= 1.9)
    for i, name in enumerate(("Left", "Right")):
        axes[0, i].plot(t[zoom], angle[zoom, i], label="True angle")
        axes[0, i].step(t[zoom], measured[zoom, i], where="post", label="Quantised angle")
        axes[0, i].set(title=f"{name} wheel: ramp zoom", ylabel="Angle [rad]")

        axes[1, i].plot(t, 1000*(measured[:, i] - angle[:, i]), lw=.7, label="Quantised - true")
        half_count = 1000*np.pi/log["settings"]["counts_per_revolution"]
        axes[1, i].axhline(half_count, color="grey", ls="--", label="Half-count bound")
        axes[1, i].axhline(-half_count, color="grey", ls="--")
        axes[1, i].set_ylabel("Angle error [mrad]")

        axes[2, i].plot(t, measured_speed[:, i], lw=.6, alpha=.7, label="Count-difference speed")
        axes[2, i].plot(t, speed[:, i], color="black", lw=1, label="True interval-average speed")
        axes[2, i].set_ylabel("Speed [rad/s]")

        axes[3, i].plot(t, measured_speed[:, i] - speed[:, i], lw=.7, label="Quantised - true")
        axes[3, i].set(ylabel="Speed error [rad/s]", xlabel="Simulation time [s]")

    for ax in axes.flat:
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)

    return figure


def plot_odometry_comparison(logs):

    figure, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    figure.suptitle("Encoder odometry vs body truth\n"
                   "Each column is a separate feedback trial; both odometries use that trial's motion")
    for column, log in enumerate(logs):
        truth = np.asarray(log["truth_pose"])
        axes[0, column].plot(*truth[:, :2].T, color="black", label="Body truth")
        for name, color in (("ideal", "tab:blue"), ("quantised", "tab:orange")):
            pose = np.asarray(log[f"odometry_{name}"])
            error = pose - truth
            error[:, 2] = wrap(error[:, 2])
            axes[0, column].plot(*pose[:, :2].T, color=color, ls="--", label=f"{name} encoder odometry")
            axes[1, column].plot(log["time"], 100*np.linalg.norm(error[:, :2], axis=1),
                                 color=color, label=name)
            axes[2, column].plot(log["time"], np.rad2deg(error[:, 2]), color=color, label=name)

        axes[0, column].set(title=f"{log['settings']['feedback']} PI feedback", xlabel="World x [m]",
                            ylabel="World y [m]", aspect="equal")
        axes[1, column].set(ylabel="Position error [cm]", xlabel="Simulation time [s]")
        axes[2, column].set(ylabel="Heading error [deg]", xlabel="Simulation time [s]")

    for ax in axes.flat:
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)

    return figure


def save_results(logs, show=True, results_folder=RESULTS_FOLDER):

    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    index = 1 + max((int(p.name[4:]) for p in results_folder.glob("run_*")
                     if p.is_dir() and p.name[4:].isdigit()), default=0)
    while True:
        folder = results_folder / f"run_{index:03d}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            index += 1

    figures = []
    summary = {}
    for log in logs:
        name = log["settings"]["feedback"]
        metrics = encoder_metrics(log)
        summary[name] = metrics

        # Reuse the existing PI plotting function without changing its callers.
        pi_figure = plot_pi_results(log, metrics)
        pi_figure.suptitle(
            f"{name.capitalize()} encoder feedback | "
            f"speed window {1000*log['settings']['speed_window']:g} ms | "
            f"PI interval {1000*log['settings']['control_dt']:g} ms\n"
            + pi_figure.get_suptitle())
        for column, color in enumerate(("tab:green", "tab:red")):
            pi_figure.axes[0].plot(log["time"], np.asarray(log["speed_truth_instant"])[:, column],
                                   color=color, lw=.7, alpha=.8,
                                   label=f"{'Left' if column == 0 else 'Right'} joint truth (instantaneous)")
        pi_figure.axes[0].legend(fontsize=9, ncol=2)
        readings_figure = plot_encoder_readings(log)
        for figure, label in ((pi_figure, "pi"), (readings_figure, "readings")):
            figure.savefig(folder / f"{label}_{name}_{index:03d}.png", dpi=150)
            figures.append(figure)

        arrays = {key: np.asarray(value) for key, value in log.items()
                  if key not in ("wheel", "settings")}
        arrays.update({f"wheel_{key}": np.asarray(value) for key, value in log["wheel"].items()})
        arrays["settings"] = np.array(json.dumps(log["settings"]))
        np.savez_compressed(folder / f"encoder_{name}_{index:03d}.npz", **arrays)

        print(f"{name}: true wheel-speed tracking RMSE L/R "
              f"{np.round(metrics['true_wheel_tracking_rmse_rad_s'], 4)} rad/s; "
              f"feedback speed-error RMSE L/R "
              f"{np.round(metrics['feedback_speed_measurement_error_rmse_rad_s'], 4)} rad/s")

    odometry_figure = plot_odometry_comparison(logs)
    odometry_figure.savefig(folder / f"odometry_comparison_{index:03d}.png", dpi=150)
    figures.append(odometry_figure)
    (folder / f"metrics_{index:03d}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved encoder diagnostic: {folder}")

    if show:
        plt.show()
    for figure in figures:
        plt.close(figure)

    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save graphs without opening plot windows.")
    parser.add_argument("--viewer", action="store_true", help="Watch each of the two simulation trials.")
    parser.add_argument("--counts-per-revolution", type=int, default=COUNTS_PER_REVOLUTION,
                        help="Encoder resolution; default 4096 counts/revolution.")
    parser.add_argument("--speed-window", type=float, default=SPEED_WINDOW,
                        help="Rolling speed window in seconds; default 0.02.")
    parser.add_argument("--control-dt", type=float, default=PI_DT,
                        help="PI update interval in seconds; default 0.01.")
    args = parser.parse_args()

    # Validate resolution before starting either simulation.
    encoder_angle_measurement(0.0, args.counts_per_revolution)
    logs = [run_encoder_test(quantised, args.counts_per_revolution, args.viewer,
                             args.speed_window, args.control_dt)
            for quantised in (False, True)]
    save_results(logs, show=not args.no_show)

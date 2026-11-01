"""Reusable plots for wheel targets and measured encoder speeds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURES_FOLDER = Path(__file__).resolve().parent / "figures"


def next_run_index():
    """Continue numbering after the highest saved graph, including partial runs."""
    indices = []

    for path in FIGURES_FOLDER.glob("*.png"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            indices.append(int(suffix))

    return max(indices, default=0) + 1


def _save_figure(figure, name, run_index):
    """Keep older results; all plots from one test share the same run number."""
    FIGURES_FOLDER.mkdir(exist_ok=True)
    if run_index is None:
        run_index = next_run_index()

    output_path = FIGURES_FOLDER / f"{name}_{run_index:03d}.png"

    # Exclusive creation also protects an old graph if a number is reused.
    with output_path.open("xb") as output_file:
        figure.savefig(output_file, format="png", dpi=150)

    print(f"Saved graph: {output_path}")


def _pi_label(log):
    """Use the settings recorded with the run, if they are available."""
    gains = log.get("pi_gains")
    if gains is None:
        return ""

    return f"\nPI control: Kp = {gains['kp']:g}    Ki = {gains['ki']:g}"


def plot_wheel_logs(log, run_index=None):
    """Plot a dictionary of lists/arrays; time is seconds, speeds are rad/s.

    Required keys: time, left_target, right_target, left_measured,
    right_measured. Call plt.show() afterwards to display the figure.
    Saves figures/wheel_logs_NNN.png beside this file, preserving older graphs.
    Optional left_torque/right_torque keys add a torque row (N*m).
    Optional pi_gains metadata adds the run's Kp and Ki to the title.
    """
    time = np.asarray(log["time"])
    has_torque = "left_torque" in log and "right_torque" in log
    rows = 3 if has_torque else 2
    figure, axes = plt.subplots(rows, 2, sharex=True, figsize=(10, 3 * rows))

    for column, wheel in enumerate(("left", "right")):
        target = np.asarray(log[f"{wheel}_target"])
        measured = np.asarray(log[f"{wheel}_measured"])

        axes[0, column].step(time, target, where="post", label="Target")
        axes[0, column].plot(time, measured, label="Measured")
        axes[0, column].set_title(f"{wheel.capitalize()} wheel")
        axes[0, column].set_ylabel("Speed (rad/s)")
        axes[0, column].legend()

        axes[1, column].plot(time, target - measured)
        axes[1, column].axhline(0, color="grey", linewidth=0.8)
        axes[1, column].set_ylabel("Target - measured (rad/s)")

        # The raw request can exceed the limit; the applied command is clipped.
        # Showing both makes saturation visible while tuning the wheel loop.
        if has_torque:
            if f"{wheel}_raw_torque" in log:
                axes[2, column].plot(
                    time, log[f"{wheel}_raw_torque"], linestyle="--", label="Requested"
                )

            axes[2, column].plot(time, log[f"{wheel}_torque"], label="Limited command")
            axes[2, column].set_ylabel("Torque (N m)")
            axes[2, column].legend()

        axes[-1, column].set_xlabel("Time (s)")

    for axis in axes.flat:
        axis.grid(True)
    figure.suptitle("Wheel speed response" + _pi_label(log))
    figure.tight_layout()

    _save_figure(figure, "wheel_logs", run_index)

    return figure


def plot_encoder_positions(log, wheel_radius, wheel_base, run_index=None):
    """Compare wheel-only odometry with diagnostic truth; lengths are metres.

    Requires time, left_angle, right_angle, x, y and heading in the log.
    Wheel angles and heading are radians. Only the INITIAL true pose is used
    to align the two trajectories; later truth never corrects the odometry.
    """
    time = np.asarray(log["time"])
    true_x = np.asarray(log["x"])
    true_y = np.asarray(log["y"])

    # Wheel rotation predicts ground travel only under the no-slip assumption.
    left_travel = wheel_radius * np.diff(log["left_angle"])
    right_travel = wheel_radius * np.diff(log["right_angle"])

    encoder_x = np.zeros(len(time))
    encoder_y = np.zeros(len(time))
    encoder_x[0], encoder_y[0] = true_x[0], true_y[0]
    heading = log["heading"][0]

    for index, (left, right) in enumerate(zip(left_travel, right_travel)):
        distance = (left + right) / 2
        heading_change = (right - left) / wheel_base

        # Midpoint integration approximates the short arc between samples.
        # Heading is propagated from encoders, even if the robot veers slightly.
        midpoint_heading = heading + heading_change / 2
        encoder_x[index + 1] = encoder_x[index] + distance * np.cos(midpoint_heading)
        encoder_y[index + 1] = encoder_y[index] + distance * np.sin(midpoint_heading)
        heading += heading_change

    # This is accumulated position mismatch, not a direct measurement of slip.
    # Radius/geometry errors and model assumptions can also produce divergence.
    position_error = np.hypot(encoder_x - true_x, encoder_y - true_y)

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    for axis, encoder, truth, coordinate in zip(
        axes[:2], (encoder_x, encoder_y), (true_x, true_y), ("x", "y")
    ):
        axis.plot(time, encoder, label="Encoder prediction", linestyle="--")
        axis.plot(time, truth, label="Ground truth")
        axis.set_ylabel(f"World {coordinate} (m)")
        axis.legend()

    axes[2].plot(time, position_error)
    axes[2].set_ylabel("Position mismatch (m)")
    axes[2].set_xlabel("Time (s)")

    for axis in axes:
        axis.grid(True)

    figure.suptitle("Encoder position versus ground truth (slip diagnostic)" + _pi_label(log))
    figure.tight_layout()

    _save_figure(figure, "encoder_positions", run_index)

    return figure


def plot_rotation_logs(log, wheel_radius, wheel_base, run_index=None):
    """Separate wheel tracking from rolling-model mismatch in a rotation test.

    Truth is used for diagnostic comparison only. Heading predictions share
    the initial true heading, with no later correction from simulator truth.
    """
    time = np.asarray(log["time"])
    target_yaw = wheel_radius / wheel_base * (
        np.asarray(log["right_target"]) - np.asarray(log["left_target"])
    )
    encoder_yaw = wheel_radius / wheel_base * (
        np.asarray(log["right_measured"]) - np.asarray(log["left_measured"])
    )
    actual_yaw = np.asarray(log["ground_yaw_rate"])

    # Commands are held over each interval; encoder ANGLE increments give
    # accumulated rotation without introducing velocity-integration error.
    commanded_heading = np.r_[0.0, np.cumsum(target_yaw[:-1] * np.diff(time))]
    left_angle = np.asarray(log["left_angle"])
    right_angle = np.asarray(log["right_angle"])
    encoder_heading = wheel_radius / wheel_base * (
        right_angle - right_angle[0] - (left_angle - left_angle[0])
    )
    actual_heading = np.unwrap(log["heading"])
    actual_heading -= actual_heading[0]
    drift = np.hypot(
        np.asarray(log["x"]) - log["x"][0],
        np.asarray(log["y"]) - log["y"][0],
    )

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))
    axes[0].plot(time, target_yaw, label="Command")
    axes[0].plot(time, encoder_yaw, "--", label="Measured-wheel prediction")
    axes[0].plot(time, actual_yaw, label="Body truth")
    axes[0].set_ylabel("Yaw rate (rad/s)")

    axes[1].plot(time, target_yaw - encoder_yaw, label="Command - wheel prediction")
    axes[1].plot(time, encoder_yaw - actual_yaw, label="Wheel prediction - body truth")
    axes[1].set_ylabel("Yaw-rate error (rad/s)")

    axes[2].plot(time, commanded_heading, label="Integrated command")
    axes[2].plot(time, encoder_heading, "--", label="Encoder prediction")
    axes[2].plot(time, actual_heading, label="Body truth (unwrapped)")
    axes[2].set_ylabel("Heading change (rad)")

    axes[3].plot(time, drift, label="Distance from initial XY position")
    axes[3].set_ylabel("Position drift (m)")
    axes[3].set_xlabel("Time (s)")

    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Rotation: command, wheel prediction and body motion" + _pi_label(log))
    figure.tight_layout()
    _save_figure(figure, "rotation_logs", run_index)
    return figure


def plot_wheel_slip(log, wheel_radius, wheel_base, run_index=None):
    """Plot each wheel's instantaneous longitudinal slip estimate in m/s.

    Uses measured wheel speeds and diagnostic ground_forward_speed (m/s)
    and ground_yaw_rate (rad/s). Assumes planar motion on a stationary floor.
    """
    time = np.asarray(log["time"])
    forward_speed = np.asarray(log["ground_forward_speed"])
    turn_rate = np.asarray(log["ground_yaw_rate"])

    # When turning, the inner wheel travels more slowly than the outer wheel.
    # Compare each wheel with ground motion on ITS side of the axle, rather
    # than incorrectly counting the normal speed difference in a turn as slip.
    left_ground_speed = forward_speed - turn_rate * wheel_base / 2
    right_ground_speed = forward_speed + turn_rate * wheel_base / 2

    # Convert encoder rotation to tyre surface speed so both sides are m/s.
    # Positive mismatch means the wheel surface is moving faster forward than
    # the chassis at that wheel; negative mismatch can occur during braking.
    left_slip = wheel_radius * np.asarray(log["left_measured"]) - left_ground_speed
    right_slip = wheel_radius * np.asarray(log["right_measured"]) - right_ground_speed

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time, left_slip, label="Left wheel", linewidth=1.2)
    axis.plot(time, right_slip, label="Right wheel", linestyle="--", linewidth=1.2)
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Wheel surface speed - ground speed (m/s)")
    axis.set_title("Instantaneous longitudinal slip estimate" + _pi_label(log))
    axis.legend()
    axis.grid(True)
    figure.tight_layout()

    _save_figure(figure, "wheel_slip", run_index)

    return figure

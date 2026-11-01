"""Arc-model diagnostics; simulator truth is used only for evaluation."""

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .plot_wheel_logs import _pi_label, _save_figure
else:
    from plot_wheel_logs import _pi_label, _save_figure


def integrate_arc(distance, heading_change, initial_pose):
    """Propagate planar pose from interval travel and turn, in metres/radians.

    The exact constant-curvature step also handles straight motion. np.sinc
    uses pi-scaled arguments, so sinc(dtheta / (2*pi)) means sin(dtheta/2)
    divided by dtheta/2, with a safe limit of one at zero turn.
    """
    pose = np.empty((len(distance) + 1, 3))
    pose[0] = initial_pose
    for index, (travel, turn) in enumerate(zip(distance, heading_change)):
        x, y, heading = pose[index]
        chord = travel * np.sinc(turn / (2 * np.pi))
        midpoint = heading + turn / 2
        pose[index + 1] = (
            x + chord * np.cos(midpoint),
            y + chord * np.sin(midpoint),
            heading + turn,
        )
    return pose


def plot_arc_logs(log, wheel_radius, wheel_base, run_index=None):
    """Plot commanded, encoder-predicted and true arcs; return summary metrics."""
    time = np.asarray(log["time"])
    dt = np.diff(time)
    truth = np.column_stack((log["x"], log["y"], np.unwrap(log["heading"])))

    left_target = np.asarray(log["left_target"])
    right_target = np.asarray(log["right_target"])
    target_v = wheel_radius * (right_target + left_target) / 2
    target_w = wheel_radius * (right_target - left_target) / wheel_base
    encoder_v = wheel_radius * (
        np.asarray(log["right_measured"]) + np.asarray(log["left_measured"])
    ) / 2
    encoder_w = wheel_radius * (
        np.asarray(log["right_measured"]) - np.asarray(log["left_measured"])
    ) / wheel_base

    # Each command is held until the next sample. Encoder increments describe
    # actual wheel travel over that same interval, independent of speed lag.
    command = integrate_arc(target_v[:-1] * dt, target_w[:-1] * dt, truth[0])
    left_travel = wheel_radius * np.diff(log["left_angle"])
    right_travel = wheel_radius * np.diff(log["right_angle"])
    encoder = integrate_arc(
        (left_travel + right_travel) / 2,
        (right_travel - left_travel) / wheel_base,
        truth[0],
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    for pose, label, style in (
        (command, "Integrated command", "--"),
        (encoder, "Encoder prediction", ":"),
        (truth, "Body truth", "-"),
    ):
        axis.plot(pose[:, 0], pose[:, 1], style, label=label)
        axis.plot(pose[-1, 0], pose[-1, 1], "o")
    axis.plot(*truth[0, :2], "kx", label="Start")
    axis.set(xlabel="World X (m)", ylabel="World Y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend()
    axis.set_title("Arc trajectories (dots mark final positions)" + _pi_label(log))
    figure.tight_layout()
    _save_figure(figure, "arc_xy", run_index)

    command_error = np.linalg.norm(command[:, :2] - truth[:, :2], axis=1)
    model_error = np.linalg.norm(encoder[:, :2] - truth[:, :2], axis=1)
    # Wrapped residuals avoid artificial jumps at the +/-pi boundary.
    command_heading_error = np.arctan2(
        np.sin(command[:, 2] - truth[:, 2]), np.cos(command[:, 2] - truth[:, 2])
    )
    model_heading_error = np.arctan2(
        np.sin(encoder[:, 2] - truth[:, 2]), np.cos(encoder[:, 2] - truth[:, 2])
    )

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))
    for values, label, style in (
        ((target_v, target_w), "Command", "--"),
        ((encoder_v, encoder_w), "Measured-wheel prediction", ":"),
        ((log["ground_forward_speed"], log["ground_yaw_rate"]), "Body truth", "-"),
    ):
        axes[0].plot(time, values[0], style, label=label)
        axes[1].plot(time, values[1], style, label=label)
    axes[0].set_ylabel("Forward speed (m/s)")
    axes[1].set_ylabel("Yaw rate (rad/s)")
    axes[2].plot(time, command_error, label="Command versus truth")
    axes[2].plot(time, model_error, label="Encoder versus truth")
    axes[2].set_ylabel("Position error (m)")
    axes[3].plot(time, command_heading_error, label="Command - truth")
    axes[3].plot(time, model_heading_error, label="Encoder - truth")
    axes[3].set_ylabel("Heading error (rad)")
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle("Arc motion and model errors" + _pi_label(log))
    figure.tight_layout()
    _save_figure(figure, "arc_metrics", run_index)

    return {
        "Final command position error (m)": command_error[-1],
        "Maximum command position error (m)": np.max(command_error),
        "Final encoder position error (m)": model_error[-1],
        "Final command heading error (rad)": command_heading_error[-1],
        "Final encoder heading error (rad)": model_heading_error[-1],
        "Body forward-speed RMSE (m/s)": np.sqrt(np.mean(
            (target_v - np.asarray(log["ground_forward_speed"]))**2
        )),
        "Body yaw-rate RMSE (rad/s)": np.sqrt(np.mean(
            (target_w - np.asarray(log["ground_yaw_rate"]))**2
        )),
    }

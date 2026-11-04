"""Wheel PI report: three overlaid plots, separate from path-controller plots."""

import matplotlib.pyplot as plt
import numpy as np


def plot_pi_results(log, metrics):
    """Compare left/right speed, speed error and applied torque at physics rate.

    Applied effort is plotted on its actual torque scale. Clipped requests are
    marked at the applied value; raw request peaks are reported numerically.
    """
    wheel = {key: np.asarray(value) for key, value in log["wheel"].items()}
    times = wheel["time"]
    settings = log["settings"]
    gains = settings["pi_gains"]

    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                layout="constrained")
    figure.suptitle(
        "Wheel PI response - left/right comparison\n"
        f"Left Kp={gains[0]['kp']:g}, Ki={gains[0]['ki']:g} | "
        f"Right Kp={gains[1]['kp']:g}, Ki={gains[1]['ki']:g}\n"
        f"Speed-error RMS L/R: {metrics['wheel_speed_rms_error_rad_s'][0]:.3f} / "
        f"{metrics['wheel_speed_rms_error_rad_s'][1]:.3f} rad/s | "
        f"Torque clipped L/R: {metrics['torque_clipped_percent'][0]:.2f} / "
        f"{metrics['torque_clipped_percent'][1]:.2f}%"
    )

    for column, (name, color) in enumerate((
        ("Left", "tab:blue"), ("Right", "tab:orange"),
    )):
        axes[0].plot(times, wheel["target"][:, column], "--", color=color,
                     label=f"{name} target", linewidth=1.5)
        axes[0].plot(times, wheel["measured"][:, column], color=color,
                     label=f"{name} encoder", linewidth=0.9)

        error = wheel["target"][:, column] - wheel["measured"][:, column]
        axes[1].plot(times, error, color=color, label=name, linewidth=0.9)

        axes[2].plot(times, wheel["torque"][:, column], color=color,
                     label=f"{name} applied", linewidth=0.9)

        clipped = np.abs(
            wheel["raw_torque"][:, column] - wheel["torque"][:, column]
        ) > 1e-9
        axes[2].scatter(times[clipped], wheel["torque"][clipped, column],
                        color=color, marker="x", s=14, label=f"{name} clipped")

        lower, upper = settings["torque_limits"][column]
        # Distinct styles also support different left/right torque limits.
        for limit in (lower, upper):
            axes[2].axhline(limit, color=color, linestyle=":", linewidth=1)

    axes[0].set(title="Wheel-speed tracking", ylabel="Wheel speed [rad/s]")
    axes[1].set(title="Target minus measured wheel speed", ylabel="Error [rad/s]")

    raw_peaks = np.max(np.abs(wheel["raw_torque"]), axis=0)
    axes[2].set(
        title=(
            "Applied actuator torque (crosses indicate PI request clipping)\n"
            f"Full-run peak absolute PI request L/R: "
            f"{raw_peaks[0]:.3f} / {raw_peaks[1]:.3f} N m; "
            f"applied RMS: {metrics['torque_rms_Nm'][0]:.3f} / "
            f"{metrics['torque_rms_Nm'][1]:.3f} N m"
        ),
        ylabel="Torque [N m]",
    )

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0, log["duration"])
        axis.legend(fontsize=9, ncol=2)
        if log["stop_time"] is not None:
            axis.axvline(log["stop_time"], color="grey", linestyle=":")

    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[-1].set_xlabel("Simulation time [s]")
    return figure

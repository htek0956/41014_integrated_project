"""Test a smooth yaw-rate ramp, two-second hold, and ramp back to rest.

Run from the project folder: python testing/motor_control_loop/rotation_test.py
Runs straight to the plots; use --viewer to watch the robot as well.
Simulator truth is logged only for this diagnostic experiment.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# Direct execution puts this test folder on the import path. Add the root so
# this experiment can reuse the same wheel controller as the future system.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.wheel_controller import WheelPI

if __package__:
    from .plot_wheel_logs import next_run_index, plot_rotation_logs, plot_wheel_logs, plot_wheel_slip
else:
    from plot_wheel_logs import next_run_index, plot_rotation_logs, plot_wheel_logs, plot_wheel_slip


WHEEL_RADIUS = 0.033  # metres
WHEEL_BASE = 0.288  # metres between the two drive wheels

# Selected PI gains: Kp=0.4 and Ki=0.5. Keep the master/ratio tuning style.
# I_RATIO is Ki/Kp; set it to zero for a P-only comparison.
MASTER_GAIN = 1
I_RATIO = 1

KP = MASTER_GAIN
KI = MASTER_GAIN * I_RATIO

# Positive body yaw is counterclockwise; all times are simulated seconds.
MAX_YAW_RATE = 0.5  # body rad/s, NOT wheel speed
RAMP_START = 1.0
RAMP_DURATION = 2.0
HOLD_DURATION = 2.0
RAMP_DOWN_START = RAMP_START + RAMP_DURATION + HOLD_DURATION
RAMP_END = RAMP_DOWN_START + RAMP_DURATION
TEST_DURATION = RAMP_END + 2.0  # Observe residual motion after returning to zero.


def yaw_target(sim_time):
    """Smoothly rise, hold, then return to zero without reversing direction."""
    if sim_time < RAMP_START:
        return 0.0

    if sim_time < RAMP_START + RAMP_DURATION:
        progress = (sim_time - RAMP_START) / RAMP_DURATION
        return MAX_YAW_RATE * (3 * progress**2 - 2 * progress**3)

    if sim_time < RAMP_DOWN_START:
        return MAX_YAW_RATE

    if sim_time < RAMP_END:
        progress = (sim_time - RAMP_DOWN_START) / RAMP_DURATION
        return MAX_YAW_RATE * (1 - 3 * progress**2 + 2 * progress**3)

    return 0.0


def wheel_targets(sim_time):
    """Zero forward speed: equal opposite wheel speeds, in wheel rad/s."""
    speed = yaw_target(sim_time) * WHEEL_BASE / (2 * WHEEL_RADIUS)
    return -speed, speed

def run_rotation_test(show_viewer=False, kp=KP, ki=KI):
    """Return measurement lists plus the PI settings used for this run."""
    scene = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    wheel_names = ("wheel_left", "wheel_right")
    actuators = [model.actuator(name).id for name in wheel_names]
    # Joint position and velocity have different indices for a free-base robot.
    position_indices = [int(model.joint(name).qposadr[0]) for name in wheel_names]
    velocity_indices = [int(model.joint(name).dofadr[0]) for name in wheel_names]

    # Each wheel must remember its own speed error. Match the software limits
    # to the motor actuator limits, avoiding hidden additional XML clipping.
    wheel_controllers = [
        WheelPI(kp, ki, model.actuator_ctrlrange[actuator])
        for actuator in actuators
    ]
    dt = float(model.opt.timestep)

    log = {key: [] for key in (
        "time", "left_target", "right_target", "left_measured", "right_measured",
        "left_angle", "right_angle", "x", "y", "heading",
        "left_torque", "right_torque", "left_raw_torque", "right_raw_torque",
        "ground_forward_speed", "ground_yaw_rate",
    )}

    # Keep the actual run settings with its data, so plots don't accidentally
    # show different gains after we edit the defaults or compare older logs.
    log["pi_gains"] = {"kp": float(kp), "ki": float(ki)}

    viewer_context = mujoco.viewer.launch_passive(model, data) if show_viewer else nullcontext(None)

    with viewer_context as viewer:
        while data.time < TEST_DURATION:
            if viewer is not None and not viewer.is_running():
                break
            frame_start = time.perf_counter()
            frame_sim_start = data.time

            # Refresh the target every physics step, including during the ramp.
            # Only the viewer refresh is grouped into batches of ten steps.
            for _ in range(10):
                if data.time >= TEST_DURATION:
                    break

                targets = wheel_targets(data.time)

                # The reference is still wheel speed in rad/s. Only the PI
                # output goes to data.ctrl, which now expects torque in N*m.
                torques = [
                    controller.update(target, float(data.qvel[velocity]), dt)
                    for controller, target, velocity in zip(wheel_controllers, targets, velocity_indices)
                ]
                data.ctrl[actuators] = torques

                # Values in each row describe the same instant, before stepping.
                log["time"].append(float(data.time))
                for wheel, target, position, velocity in zip(("left", "right"), targets, position_indices, velocity_indices):
                    log[f"{wheel}_target"].append(target)
                    log[f"{wheel}_measured"].append(float(data.qvel[velocity]))
                    log[f"{wheel}_angle"].append(float(data.qpos[position]))

                for wheel, controller, torque in zip(
                    ("left", "right"), wheel_controllers, torques
                ):
                    log[f"{wheel}_torque"].append(float(torque))
                    log[f"{wheel}_raw_torque"].append(float(controller.raw_torque))

                log["x"].append(float(data.qpos[0]))
                log["y"].append(float(data.qpos[1]))
                qw, qx, qy, qz = data.qpos[3:7]
                log["heading"].append(float(np.arctan2(
                    2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)
                )))

                # Diagnostic truth only: project world motion along the robot's
                # forward direction. Use velocity directly, avoiding noisy
                # differences between consecutive position samples.
                heading = log["heading"][-1]
                forward_speed = (
                    data.qvel[0] * np.cos(heading)
                    + data.qvel[1] * np.sin(heading)
                )
                log["ground_forward_speed"].append(float(forward_speed))

                # MuJoCo stores free-joint angular velocity in the body frame.
                # Rotate it to world coordinates before taking the turn rate.
                world_angular_velocity = np.zeros(3)
                mujoco.mju_rotVecQuat(world_angular_velocity, data.qvel[3:6], data.qpos[3:7])
                log["ground_yaw_rate"].append(float(world_angular_velocity[2]))

                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                # Pace the viewer using simulated time, not a fixed wall-time run.
                time.sleep(max(0.0, data.time - frame_sim_start - (time.perf_counter() - frame_start)))

    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    viewer_options = parser.add_mutually_exclusive_group()
    viewer_options.add_argument("--viewer", action="store_true", help="Watch the robot before plotting.")
    viewer_options.add_argument("--headless", action="store_true", help="Skip the robot viewer (the default).")
    args = parser.parse_args()
    log = run_rotation_test(show_viewer=args.viewer)
    if not log["time"]:
        return

    # Save a matched graph set without overwriting previous runs.
    run_index = next_run_index()
    plot_wheel_logs(log, run_index=run_index)
    plot_rotation_logs(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)
    plot_wheel_slip(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)

    times = np.asarray(log["time"])
    actual_yaw = np.asarray(log["ground_yaw_rate"])
    encoder_yaw = WHEEL_RADIUS / WHEEL_BASE * (
        np.asarray(log["right_measured"]) - np.asarray(log["left_measured"])
    )
    heading = np.unwrap(log["heading"])
    drift = np.hypot(
        np.asarray(log["x"]) - log["x"][0],
        np.asarray(log["y"]) - log["y"][0],
    )
    hold = (times >= RAMP_START + RAMP_DURATION) & (times < RAMP_DOWN_START)
    print(f"Recorded {times[-1]:.3f} simulated seconds (full test: {TEST_DURATION:.1f} s).")
    print(f"Ideal full-profile heading change: {MAX_YAW_RATE * (RAMP_DURATION + HOLD_DURATION):.4f} rad")
    print(f"Actual heading change: {heading[-1] - heading[0]:.4f} rad")
    print(f"Maximum position drift: {np.max(drift):.5f} m")
    print(f"Final body yaw rate: {actual_yaw[-1]:.5f} rad/s")
    if np.any(hold):
        print(f"Hold yaw tracking RMSE: {np.sqrt(np.mean((MAX_YAW_RATE - actual_yaw[hold])**2)):.5f} rad/s")
        print(f"Hold rolling-model RMSE: {np.sqrt(np.mean((encoder_yaw[hold] - actual_yaw[hold])**2)):.5f} rad/s")

    plt.show()


if __name__ == "__main__":
    main()


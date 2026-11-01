"""Test a smooth speed increase, an abrupt increase, and an abrupt stop.

Run from the project folder: python testing/motor_control_loop/motion_test.py
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
    from .plot_wheel_logs import next_run_index, plot_encoder_positions, plot_wheel_logs, plot_wheel_slip
else:
    from plot_wheel_logs import next_run_index, plot_encoder_positions, plot_wheel_logs, plot_wheel_slip


WHEEL_RADIUS = 0.033  # metres
WHEEL_BASE = 0.288  # metres between the two drive wheels

# Selected PI gains: Kp=0.4 and Ki=0.5. Keep the master/ratio tuning style.
# I_RATIO is Ki/Kp; set it to zero for a P-only comparison.
MASTER_GAIN = 1
I_RATIO = 1

KP = MASTER_GAIN
KI = MASTER_GAIN * I_RATIO

# Both wheels follow the same profile so the robot should travel straight.
# Wheel speeds are rad/s; all times below are simulated seconds.
LOW_SPEED = 4.0
HIGH_SPEED = 6.0

RAMP_START = 1.0
RAMP_DURATION = 2.0
STEP_UP_TIME = 5.0
STEP_DOWN_TIME = 8.0
TEST_DURATION = 11.0


def wheel_target(sim_time):
    """Return the wheel-speed target at this point in the experiment."""
    if sim_time < RAMP_START:
        return 0.0

    if sim_time < RAMP_START + RAMP_DURATION:
        # Smoothstep rises from 0 to 1 with zero slope at both ends.
        # This avoids an abrupt change in target acceleration at the joins.
        progress = (sim_time - RAMP_START) / RAMP_DURATION
        smooth_progress = 3 * progress**2 - 2 * progress**3
        return LOW_SPEED * smooth_progress

    if sim_time < STEP_UP_TIME:
        return LOW_SPEED

    if sim_time < STEP_DOWN_TIME:
        # No ramp here: deliberately test the response to a sudden increase.
        return HIGH_SPEED

    # A sudden zero target tests braking; keep simulating to observe settling.
    return 0.0


def run_motion_test(show_viewer=False, kp=KP, ki=KI):
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

                target = wheel_target(data.time)

                # The reference is still wheel speed in rad/s. Only the PI
                # output goes to data.ctrl, which now expects torque in N*m.
                torques = [
                    controller.update(target, float(data.qvel[velocity]), dt)
                    for controller, velocity in zip(wheel_controllers, velocity_indices)
                ]
                data.ctrl[actuators] = torques

                # Values in each row describe the same instant, before stepping.
                log["time"].append(float(data.time))
                for wheel, position, velocity in zip(("left", "right"), position_indices, velocity_indices):
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
    log = run_motion_test(show_viewer=args.viewer)
    if not log["time"]:
        return

    encoder_travel = WHEEL_RADIUS * (
        log["left_angle"][-1] - log["left_angle"][0]
        + log["right_angle"][-1] - log["right_angle"][0]
    ) / 2
    displacement = np.hypot(log["x"][-1] - log["x"][0], log["y"][-1] - log["y"][0])

    # Integrate the recorded target over time instead of assuming one speed.
    # Each target was applied over the interval leading to the next sample.
    target_travel = WHEEL_RADIUS * np.sum(
        np.asarray(log["left_target"][:-1]) * np.diff(log["time"])
    )

    print(f"Recorded {log['time'][-1]:.3f} simulated seconds (full test: {TEST_DURATION:.1f} s).")
    print(f"Ideal travel from recorded targets: {target_travel:.3f} m")
    print(f"Encoder travel, including stopping: {encoder_travel:.3f} m")
    print(f"True net displacement, including stopping: {displacement:.3f} m")
    # Give this test one new number so its three graphs stay together.
    run_index = next_run_index()
    plot_wheel_logs(log, run_index=run_index)

    # Compare measured wheel rotation with actual movement to investigate slip.
    # Ground truth is used for diagnostics, never as an odometry correction.
    plot_encoder_positions(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)
    plot_wheel_slip(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)

    plt.show()


if __name__ == "__main__":
    main()

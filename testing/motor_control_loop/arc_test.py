import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.wheel_controller import WheelPI

if __package__:
    from .plot_wheel_logs import next_run_index, plot_wheel_logs
    from .plot_arc_logs import plot_arc_logs
else:
    from plot_wheel_logs import next_run_index, plot_wheel_logs
    from plot_arc_logs import plot_arc_logs


WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.288


MASTER_GAIN = 1
I_RATIO = 1

KP = MASTER_GAIN
KI = MASTER_GAIN * I_RATIO


FORWARD_SPEED = 0.10
ARC_RADIUS = 0.50
MAX_YAW_RATE = FORWARD_SPEED / ARC_RADIUS
RAMP_START = 1.0
RAMP_DURATION = 2.0
HOLD_DURATION = 8.0
RAMP_DOWN_START = RAMP_START + RAMP_DURATION + HOLD_DURATION
RAMP_END = RAMP_DOWN_START + RAMP_DURATION
TEST_DURATION = RAMP_END + 2.0


def yaw_target(sim_time):

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

    yaw_rate = yaw_target(sim_time)
    forward_speed = ARC_RADIUS * yaw_rate
    left = (forward_speed - WHEEL_BASE * yaw_rate / 2) / WHEEL_RADIUS
    right = (forward_speed + WHEEL_BASE * yaw_rate / 2) / WHEEL_RADIUS
    return left, right

def run_arc_test(show_viewer=False, kp=KP, ki=KI):

    scene = PROJECT_ROOT / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    wheel_names = ("wheel_left", "wheel_right")
    actuators = [model.actuator(name).id for name in wheel_names]

    position_indices = [int(model.joint(name).qposadr[0]) for name in wheel_names]
    velocity_indices = [int(model.joint(name).dofadr[0]) for name in wheel_names]


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


    log["pi_gains"] = {"kp": float(kp), "ki": float(ki)}

    viewer_context = mujoco.viewer.launch_passive(model, data) if show_viewer else nullcontext(None)

    with viewer_context as viewer:
        while data.time < TEST_DURATION:
            if viewer is not None and not viewer.is_running():
                break
            frame_start = time.perf_counter()
            frame_sim_start = data.time


            for _ in range(10):
                if data.time >= TEST_DURATION:
                    break

                targets = wheel_targets(data.time)


                torques = [
                    controller.update(target, float(data.qvel[velocity]), dt)
                    for controller, target, velocity in zip(wheel_controllers, targets, velocity_indices)
                ]
                data.ctrl[actuators] = torques


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


                heading = log["heading"][-1]
                forward_speed = (
                    data.qvel[0] * np.cos(heading)
                    + data.qvel[1] * np.sin(heading)
                )
                log["ground_forward_speed"].append(float(forward_speed))


                world_angular_velocity = np.zeros(3)
                mujoco.mju_rotVecQuat(world_angular_velocity, data.qvel[3:6], data.qpos[3:7])
                log["ground_yaw_rate"].append(float(world_angular_velocity[2]))

                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

                time.sleep(max(0.0, data.time - frame_sim_start - (time.perf_counter() - frame_start)))

    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    viewer_options = parser.add_mutually_exclusive_group()
    viewer_options.add_argument("--viewer", action="store_true", help="Watch the robot before plotting.")
    viewer_options.add_argument("--headless", action="store_true", help="Skip the robot viewer (the default).")
    args = parser.parse_args()
    log = run_arc_test(show_viewer=args.viewer)
    if not log["time"]:
        return


    run_index = next_run_index()
    plot_wheel_logs(log, run_index=run_index)
    metrics = plot_arc_logs(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)

    print(f"Recorded {log['time'][-1]:.3f} simulated seconds (full test: {TEST_DURATION:.1f} s).")
    print(f"Commanded arc radius: {ARC_RADIUS:.3f} m")
    print(f"Peak forward speed: {FORWARD_SPEED:.3f} m/s; peak yaw rate: {MAX_YAW_RATE:.3f} rad/s")
    for label, value in metrics.items():
        print(f"{label}: {value:.5f}")

    plt.show()


if __name__ == "__main__":
    main()



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
    from .plot_wheel_logs import next_run_index, plot_encoder_positions, plot_wheel_logs, plot_wheel_slip
else:
    from plot_wheel_logs import next_run_index, plot_encoder_positions, plot_wheel_logs, plot_wheel_slip


WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.288


MASTER_GAIN = 1
I_RATIO = 1

KP = MASTER_GAIN
KI = MASTER_GAIN * I_RATIO


LOW_SPEED = 4.0
HIGH_SPEED = 6.0

RAMP_START = 1.0
RAMP_DURATION = 2.0
STEP_UP_TIME = 5.0
STEP_DOWN_TIME = 8.0
TEST_DURATION = 11.0


def wheel_target(sim_time):

    if sim_time < RAMP_START:
        return 0.0

    if sim_time < RAMP_START + RAMP_DURATION:


        progress = (sim_time - RAMP_START) / RAMP_DURATION
        smooth_progress = 3 * progress**2 - 2 * progress**3
        return LOW_SPEED * smooth_progress

    if sim_time < STEP_UP_TIME:
        return LOW_SPEED

    if sim_time < STEP_DOWN_TIME:

        return HIGH_SPEED


    return 0.0


def run_motion_test(show_viewer=False, kp=KP, ki=KI):

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

                target = wheel_target(data.time)


                torques = [
                    controller.update(target, float(data.qvel[velocity]), dt)
                    for controller, velocity in zip(wheel_controllers, velocity_indices)
                ]
                data.ctrl[actuators] = torques


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
    log = run_motion_test(show_viewer=args.viewer)
    if not log["time"]:
        return

    encoder_travel = WHEEL_RADIUS * (
        log["left_angle"][-1] - log["left_angle"][0]
        + log["right_angle"][-1] - log["right_angle"][0]
    ) / 2
    displacement = np.hypot(log["x"][-1] - log["x"][0], log["y"][-1] - log["y"][0])


    target_travel = WHEEL_RADIUS * np.sum(
        np.asarray(log["left_target"][:-1]) * np.diff(log["time"])
    )

    print(f"Recorded {log['time'][-1]:.3f} simulated seconds (full test: {TEST_DURATION:.1f} s).")
    print(f"Ideal travel from recorded targets: {target_travel:.3f} m")
    print(f"Encoder travel, including stopping: {encoder_travel:.3f} m")
    print(f"True net displacement, including stopping: {displacement:.3f} m")

    run_index = next_run_index()
    plot_wheel_logs(log, run_index=run_index)


    plot_encoder_positions(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)
    plot_wheel_slip(log, WHEEL_RADIUS, WHEEL_BASE, run_index=run_index)

    plt.show()


if __name__ == "__main__":
    main()

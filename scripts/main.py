"""Paperclip controller diagnostic using ground-truth feedback, not the EKF.

Run scripts/main.py for the viewer, or add --headless to run without animation.
Ground truth is deliberately used to isolate controller and actuator behaviour.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from controller import Controller, tracking_errors
from paperclip_path import PaperclipReference
from robot import Robot


# Experiment settings. Geometry and controller gains stay with their owners.
# Outer tracking-controller interval [s].
# Initial value from Assignment 3; validate it on this robot.
# This is not the physics timestep or necessarily the wheel PI interval.
CONTROL_DT = 0.1


# Nominal path speed [m/s]. Start slowly while validating tracking.
# This is an initial experiment choice, not a measured robot speed limit.
REFERENCE_SPEED = 0.6


# Maximum magnitude of requested body yaw rate [rad/s].
# Provisional value carried from Assignment 3; validate against wheel limits.
# The current XML defines torque limits, so this cannot be read from ctrlrange.
# The robot interface must also limit wheel targets; this bound alone does not
# guarantee feasible wheel speeds when forward motion and turning are combined.
OMEGA_MAX = 2


# Initial estimator belief, independent of the actual simulation spawn pose.
# World-frame order: x [m], y [m], heading [rad], positive counterclockwise.
# Provisional Assignment 3 values, to match the chosen paperclip placement.
INITIAL_POSE = np.array([0.3, -2.0, np.deg2rad(10.0)])


# Initial uncertainty: standard deviations of 0.3 m, 0.3 m and 10 degrees.
# Diagonal entries are variances; initial cross-covariances are assumed zero.
# These describe starting uncertainty, not process or camera measurement noise.
INITIAL_COVARIANCE = np.diag([
    0.3**2,
    0.3**2,
    np.deg2rad(10.0)**2,
])


def main(show_viewer=True, max_time=None):
    """Run the diagnostic and return time-aligned outer-control samples."""
    scene = (
        Path(__file__).resolve().parent.parent
        / "robotis_tb3"
        / "scene_turtlebot3_waffle_pi.xml"
    )

    paperclip = PaperclipReference(REFERENCE_SPEED)
    controller = Controller(REFERENCE_SPEED, CONTROL_DT)

    # Allow substantial tracking delay, but do not run forever if a segment
    # transition or stopping condition is never reached. All times are sim time.
    if max_time is None:
        max_time = 2 * paperclip.total_length / REFERENCE_SPEED + 10

    if max_time <= 0 or CONTROL_DT <= 0:
        raise ValueError("Run duration and control interval must be positive.")

    log = {name: [] for name in (
        "time", "pose", "reference", "cross_track_error", "heading_error",
        "command", "limited_command",
    )}

    # Keep wheel/effort samples at physics rate, rather than only at LQR rate.
    # Each row describes the state and newly applied torque before mj_step.
    log["wheel"] = {name: [] for name in (
        "time", "target", "measured", "torque", "raw_torque",
        "body_command", "limited_command", "body_measured",
    )}
    log["reference_path"] = np.array([
        [point["x"], point["y"]]
        for point in (
            paperclip.reference_at(distance)
            for distance in np.linspace(0, paperclip.total_length, 1000)
        )
    ])

    robot = Robot(scene, max_yaw_rate=OMEGA_MAX, show_viewer=show_viewer)

    start_pos = paperclip.start_position + np.array([0.5, 0.5])
    start_heading = paperclip.start_heading + np.deg2rad(10.0)

    try:
        robot.reset((*start_pos, start_heading))

        log["settings"] = {
            "control_dt": CONTROL_DT,
            "physics_dt": robot.physics_dt,
            "reference_speed": REFERENCE_SPEED,
            "lqr_gain": controller.K_lqr.tolist(),
            "pi_gains": [
                {"kp": wheel.kp, "ki": wheel.ki}
                for wheel in robot.wheel_controllers
            ],
            "torque_limits": robot.model.actuator_ctrlrange[robot.actuators].tolist(),
            "max_wheel_speed": robot.max_wheel_speed,
            "max_yaw_rate": robot.max_yaw_rate,
        }

        command = (0.0, 0.0)
        next_control_time = 0.0
        next_viewer_time = 0.0
        stop_time = None

        # Pace the viewer against one fixed wall-clock origin. Sleeping once
        # per physics step would accumulate timing overhead on Windows.
        wall_start = time.perf_counter()

        print("Controller-only diagnostic: ground-truth pose feedback.")

        while robot.running() and robot.time < max_time:
            if robot.time + 1e-10 >= next_control_time:
                # Later the operational pipeline will supply an EKF estimate.
                # For this diagnostic, both reference and controller use truth.
                pose = robot.get_truth()
                reference = paperclip.update(pose)

                if reference["stop"]:
                    command = (0.0, 0.0)
                    if stop_time is None:
                        stop_time = robot.time
                else:
                    command = controller.update(
                        pose, reference, CONTROL_DT
                    )

                cross_track, heading_error = tracking_errors(pose, reference)
                log["time"].append(robot.time)
                log["pose"].append(pose)
                log["reference"].append(reference.copy())
                log["cross_track_error"].append(cross_track)
                log["heading_error"].append(heading_error)
                log["command"].append(command)

                next_control_time += CONTROL_DT
                control_updated = True
            else:
                control_updated = False

            # Inner wheel PI runs every physics step. Hold the outer body
            # command between LQR updates; apply torque before stepping physics.
            limited_command = robot.apply_body_command(command, robot.physics_dt)
            if control_updated:
                log["limited_command"].append(limited_command)

            encoders = robot.read_encoders()
            wheel_log = log["wheel"]
            wheel_log["time"].append(robot.time)
            wheel_log["target"].append(robot.wheel_targets.copy())
            wheel_log["measured"].append([
                encoders["left_speed"], encoders["right_speed"]
            ])
            wheel_log["torque"].append(robot.wheel_torques.copy())
            wheel_log["raw_torque"].append([
                wheel.raw_torque for wheel in robot.wheel_controllers
            ])
            wheel_log["body_command"].append(command)
            wheel_log["limited_command"].append(limited_command)
            wheel_log["body_measured"].append(robot.get_truth_velocity())

            robot.step()

            if show_viewer and robot.time >= next_viewer_time:
                robot.sync_viewer()
                next_viewer_time = robot.time + 1 / 30

                remaining_wall_time = (
                    wall_start + robot.time - time.perf_counter()
                )
                if remaining_wall_time > 0:
                    time.sleep(remaining_wall_time)

            # Zero speed is a PI target, not an instant physical stop. Continue
            # simulating for two seconds so braking and settling are observable.
            if stop_time is not None and robot.time - stop_time >= 2.0:
                break

        final_pose = robot.get_truth()
        endpoint_error = float(np.linalg.norm(
            np.asarray(final_pose[:2]) - paperclip.end_position
        ))

        log["completed"] = paperclip.finished
        log["final_pose"] = final_pose
        log["endpoint_error"] = endpoint_error
        log["stop_time"] = stop_time
        log["duration"] = robot.time

        if paperclip.finished:
            print("Reached the final path endpoint.")
        elif robot.time >= max_time:
            print("Simulation time limit reached before path completion.")
        else:
            print("Viewer closed before path completion.")

        print(f"Simulated time: {robot.time:.2f} s")
        print(f"Final position error: {endpoint_error:.3f} m")

        if log["cross_track_error"]:
            rms_error = np.sqrt(np.mean(np.square(log["cross_track_error"])))
            print(f"Cross-track RMS, including startup/stopping: {rms_error:.3f} m")

        return log

    finally:
        robot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="Run without viewer or pacing."
    )
    parser.add_argument(
        "--max-time", type=float, default=None,
        help="Maximum simulation duration in seconds."
    )
    args = parser.parse_args()

    log = main(show_viewer=not args.headless, max_time=args.max_time)

    from plot_controller_results import save_controller_results

    save_controller_results(log, show=not args.headless)

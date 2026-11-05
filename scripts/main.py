"""Map-planned loop diagnostic using ground-truth feedback, not the EKF.

Run scripts/main.py for the viewer, or add --headless to run without animation.
Ground truth is deliberately used to isolate controller and actuator behaviour.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from controller import Controller, tracking_errors
from path_planning import map_from_scene, PlannedLoopReference, plot_plan
from robot import Robot
import cv2


# Experiment settings. Geometry and controller gains stay with their owners.
# Disable detailed result collection and saved plots/data.
# The simulation and live camera/viewer windows still run normally.
LOG_RESULTS = False


# Outer tracking-controller interval [s].
# Initial value from Assignment 3; validate it on this robot.
# This is not the physics timestep or necessarily the wheel PI interval.
CONTROL_DT = 0.01


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
    """Run the diagnostic; return recorded samples, or None without logging."""
    time_since_viewer_update = 0
    scene = (
        Path(__file__).resolve().parent.parent
        / "robotis_tb3"
        / "scene_loop.xml"
    )

    grid, start = map_from_scene(scene)
    path = PlannedLoopReference(grid, start, REFERENCE_SPEED)
    print(f"Ground-truth diagnostic: planned {path.total_length:.1f} m loop from wall map.")
    controller = Controller(REFERENCE_SPEED, CONTROL_DT, verbose=LOG_RESULTS)

    # Allow substantial tracking delay, but do not run forever if a segment
    # transition or stopping condition is never reached. All times are sim time.
    if max_time is None:
        max_time = 5 * path.total_length / REFERENCE_SPEED + 20

    if max_time <= 0 or CONTROL_DT <= 0:
        raise ValueError("Run duration and control interval must be positive.")

    log = None
    if LOG_RESULTS:
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
                path.reference_at(distance)
                for distance in np.linspace(0, path.total_length, 1000)
            )
        ])

    robot = Robot(scene, max_yaw_rate=OMEGA_MAX, show_viewer=show_viewer)

    start_pos = path.start_position.copy()
    start_heading = path.start_heading
    plan_figure = None

    try:
        robot.reset((*start_pos, start_heading))

        if show_viewer:
            import matplotlib.pyplot as plt

            # Display the planned map/reference once, independently of logging.
            # Non-blocking: keep this figure open alongside the robot windows.
            plan_figure = plot_plan(path)
            plan_figure.canvas.manager.set_window_title('Map and planned route')
            plt.show(block=False)
            plan_figure.canvas.draw()
            plan_figure.canvas.flush_events()

        if LOG_RESULTS:
            log["settings"] = {
                "control_dt": CONTROL_DT,
                "physics_dt": robot.physics_dt,
                "reference_speed": REFERENCE_SPEED,
                "path": "map_planned_loop",
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

        if LOG_RESULTS:
            print("Controller-only diagnostic: ground-truth pose feedback.")

        while robot.running() and robot.time < max_time:
            if robot.time + 1e-10 >= next_control_time:
                # Later the operational pipeline will supply an EKF estimate.
                # For this diagnostic, both reference and controller use truth.
                pose = robot.get_truth()
                reference = path.update(pose, robot.get_truth_velocity())

                if reference["stop"]:
                    command = (0.0, 0.0)
                    if stop_time is None:
                        stop_time = robot.time
                else:
                    command = controller.update(
                        pose, reference, CONTROL_DT
                    )

                if LOG_RESULTS:
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
            if LOG_RESULTS and control_updated:
                log["limited_command"].append(limited_command)

            if show_viewer and robot.time - time_since_viewer_update > 0.2:
                camera = robot.read_camera()
                rgb = camera["rgb"]
            
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robot camera", bgr)

                # Process window events so the image updates.
                cv2.waitKey(1)

                time_since_viewer_update = robot.time




            if LOG_RESULTS:
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
                if plan_figure is not None and plt.fignum_exists(plan_figure.number):
                    # Service pan/zoom/close events without redrawing the route
                    # or coupling Matplotlib to the physics/control frequency.
                    plan_figure.canvas.flush_events()
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

        print(f"Lap completed: {path.finished}; time: {robot.time:.2f} s; "
              f"progress: {path.progress:.2f}/{path.total_length:.2f} m; "
              f"endpoint error: {np.linalg.norm(np.asarray(robot.get_truth()[:2])-path.end_position):.3f} m")
        if not LOG_RESULTS:
            return None

        final_pose = robot.get_truth()
        endpoint_error = float(np.linalg.norm(
            np.asarray(final_pose[:2]) - path.end_position
        ))

        log["completed"] = path.finished
        log["final_pose"] = final_pose
        log["endpoint_error"] = endpoint_error
        log["stop_time"] = stop_time
        log["duration"] = robot.time

        if path.finished:
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
        cv2.destroyAllWindows()
        if plan_figure is not None:
            plt.close(plan_figure)


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

    if log is not None:
        from plot_controller_results import save_controller_results

        save_controller_results(log, show=not args.headless)

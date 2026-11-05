import argparse
from pathlib import Path
import time

import numpy as np

from controller import Controller, tracking_errors
from path_planning import map_from_scene, PlannedLoopReference, plot_plan
from robot import Robot
from visual_docking import StationDetector, VisualDocking
import cv2


LOG_RESULTS = False


CONTROL_DT = 0.01


REFERENCE_SPEED = 0.6


OMEGA_MAX = 2


INITIAL_POSE = np.array([0.3, -2.0, np.deg2rad(10.0)])


INITIAL_COVARIANCE = np.diag([
    0.3**2,
    0.3**2,
    np.deg2rad(10.0)**2,
])

def main(show_viewer=True, max_time=None, camera_diagnostics=False):

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
    obstacle_ids = {i for i in range(robot.model.ngeom)
                    if (robot.model.geom(i).name or '').startswith(('nav_wall_', 'scenery_wall_', 'dock_'))}
    obstacle_contact_steps = 0

    start_pos = path.start_position.copy()
    start_heading = path.start_heading
    plan_figure = None
    camera_samples = []

    try:
        robot.reset((*start_pos, start_heading))
        detector = StationDetector(robot.model.camera('robot_camera').fovy[0])
        docking = VisualDocking()
        docking_start = None
        next_docking_time = 0.
        latest_camera = None

        if show_viewer:
            import matplotlib.pyplot as plt


            plan_figure = plot_plan(path)
            truth_marker, = plan_figure.axes[0].plot([], [], '+', color='tab:orange', markersize=12,
                                                   label='Truth (verification only)')
            camera_marker, = plan_figure.axes[0].plot([], [], 'o', color='tab:green', markersize=6,
                                                    label='Camera-only pose')
            plan_figure.axes[0].legend()
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


        wall_start = time.perf_counter()

        if LOG_RESULTS:
            print("Controller-only diagnostic: ground-truth pose feedback.")

        while robot.running() and robot.time < max_time:
            if robot.time + 1e-10 >= next_control_time:


                pose = robot.get_truth()
                reference = path.update(pose, robot.get_truth_velocity())

                if reference["stop"]:
                    if docking_start is None:
                        docking_start = robot.time
                        print('Lap complete. Acquiring charging-station marker for visual docking.')
                    if stop_time is not None:
                        command = (0., 0.)
                    elif robot.time-docking_start > 60:
                        command = (0., 0.)
                        stop_time = robot.time
                        print('Docking timed out; stopping safely without claiming success.')
                    elif robot.time + 1e-10 >= next_docking_time:


                        latest_camera = robot.read_camera()
                        observation = detector.observe(latest_camera['rgb'], latest_camera['timestamp'])
                        enc = robot.read_encoders()
                        speed = robot.wheel_radius*(enc['left_speed']+enc['right_speed'])/2
                        yaw = robot.wheel_radius*(enc['right_speed']-enc['left_speed'])/robot.wheel_separation
                        command = docking.update(observation, speed, yaw)
                        next_docking_time = robot.time + .05
                        if docking.finished:
                            stop_time = robot.time
                            print('Docking position reached at the physical connector pads (camera-based stop; no contact sensor).')
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


            limited_command = robot.apply_body_command(command, robot.physics_dt)
            if LOG_RESULTS and control_updated:
                log["limited_command"].append(limited_command)

            if (show_viewer or camera_diagnostics) and robot.time - time_since_viewer_update > 0.2:
                camera = robot.read_camera()
                measurement = robot.read_camera_pose(camera, annotate=True)
                rgb = measurement['annotated_rgb']


                comparison_truth = np.asarray(robot.get_truth())
                camera_samples.append(dict(timestamp=measurement['timestamp'],
                    pose=None if not measurement['valid'] else measurement['pose'].copy(),
                    truth=comparison_truth.copy(), ids_used=measurement['ids_used'],
                    covariance=measurement['covariance']))
                if show_viewer:
                    truth_marker.set_data([comparison_truth[0]], [comparison_truth[1]])
                if measurement['valid']:
                    estimate = measurement['pose']
                    position_error = np.linalg.norm(estimate[:2]-comparison_truth[:2])
                    heading_error = (estimate[2]-comparison_truth[2]+np.pi) % (2*np.pi)-np.pi
                    comparison = f'TRUTH CHECK ONLY: error {position_error*100:.1f}cm / {np.rad2deg(heading_error):+.2f}deg'
                    if show_viewer:
                        camera_marker.set_data([estimate[0]], [estimate[1]])
                else:
                    comparison = 'No camera measurement; no stale pose returned'
                    if show_viewer:
                        camera_marker.set_data([], [])
                cv2.rectangle(rgb, (0, rgb.shape[0]-32), (rgb.shape[1], rgb.shape[0]), (20, 20, 20), -1)
                cv2.putText(rgb, comparison, (8, rgb.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (255, 255, 255), 1, cv2.LINE_AA)
                if show_viewer and plt.fignum_exists(plan_figure.number):
                    plan_figure.canvas.draw_idle()

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if show_viewer:
                    cv2.imshow("Robot camera", bgr)


                if show_viewer:
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
            obstacle_contact_steps += int(any(int(c.geom1) in obstacle_ids or int(c.geom2) in obstacle_ids
                                               for c in robot.data.contact))

            if show_viewer and robot.time >= next_viewer_time:
                robot.sync_viewer()
                if plan_figure is not None and plt.fignum_exists(plan_figure.number):


                    plan_figure.canvas.flush_events()
                next_viewer_time = robot.time + 1 / 30

                remaining_wall_time = (
                    wall_start + robot.time - time.perf_counter()
                )
                if remaining_wall_time > 0:
                    time.sleep(remaining_wall_time)


            if stop_time is not None and robot.time - stop_time >= 2.0:
                break

        dock_target = robot.model.site('dock_target').pos[:2]
        dock_error = float(np.linalg.norm(np.asarray(robot.get_truth()[:2])-dock_target))
        print(f"Lap completed: {path.finished}; docked: {docking.finished}; time: {robot.time:.2f} s; "
              f"progress: {path.progress:.2f}/{path.total_length:.2f} m; "
              f"dock error: {dock_error:.3f} m; obstacle-contact steps: {obstacle_contact_steps}")
        if not LOG_RESULTS:
            return None

        final_pose = robot.get_truth()

        dock_target = robot.model.site('dock_target').pos[:2]
        endpoint_error = float(np.linalg.norm(np.asarray(final_pose[:2])-dock_target))

        log["completed"] = docking.finished
        log["lap_completed"] = path.finished
        log["docked"] = docking.finished
        log["dock_mode"] = docking.mode
        log["obstacle_contact_steps"] = obstacle_contact_steps
        log["final_body_velocity"] = robot.get_truth_velocity()
        log["docking_duration_s"] = None if docking_start is None else robot.time-docking_start


        log["docking_camera_rgb"] = None if latest_camera is None else latest_camera['rgb'].copy()
        log["camera_measurements"] = camera_samples
        log["final_pose"] = final_pose
        log["endpoint_error"] = endpoint_error
        log["stop_time"] = stop_time
        log["duration"] = robot.time

        if docking.finished:
            print("Lap and visual docking completed.")
        elif path.finished:
            print("Lap completed, but visual docking did not complete.")
        elif robot.time >= max_time:
            print("Simulation time limit reached before path completion.")
        else:
            print("Viewer closed before path completion.")

        print(f"Simulated time: {robot.time:.2f} s")
        print(f"Final position error: {endpoint_error:.3f} m")

        if log["cross_track_error"]:
            rms_error = np.sqrt(np.mean(np.square(log["cross_track_error"])))
            print(f"Reference cross-track RMS, including deliberate docking departure: {rms_error:.3f} m")

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

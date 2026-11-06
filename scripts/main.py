import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from controller import Controller
from path_planning import map_from_scene, PlannedLoopReference
from robot import Robot
from estimator import PoseEKF
from visual_docking import StationDetector, VisualDocking
from evaluation import RunLogger
from collision_detection import check_collision
from navigation import Navigation
from add_obstacles import randomise_room_obstacles


# Retain controller data for callers even when controller plots are not selected.
# Report selection lives in evaluation.py, or use --plots on the command line.
LOG_RESULTS = False

# Simulate loss of the EKF localisation camera; encoder prediction continues.
CAMERA_DROPOUTS = False
# (start, end) in simulation seconds. Edit these to change outage durations.
CAMERA_DROPOUT_PERIODS = ((10.0, 12.0), (25.0, 30.0), (45.0, 65.0))


# All intervals are simulation seconds. Physics continues between controller
# events, holding the last torque. The speed window is a history duration,
# not another update rate. Navigation uses the EKF pose estimate.
PHYSICS_DT = 0.002             # 500 Hz: MuJoCo integration.
ENCODER_DT = 0.002             # 500 Hz: quantised wheel-angle samples.
ENCODER_SPEED_WINDOW = 0.02    # 20 ms: rolling angle-to-speed estimate for PI.
PI_DT = 0.01                  # 100 Hz: wheel PI; hold torque between updates.
CONTROL_DT = 0.01             # 100 Hz: navigation/LQR body command.
EKF_DT = 0.01                 # 100 Hz: predict from raw encoder-angle increments.
CAMERA_DT = 0.2               # 5 Hz: new localisation image and EKF correction.
LIDAR_DT = 0.2                # 5 Hz: one 360-degree LDS-01 scan.
DOCKING_DT = 0.05             # 20 Hz: docking image and visual control.
CAMERA_DISPLAY_DT = 0.2       # 5 Hz: diagnostic image display only.
VIEWER_DT = 1 / 30            # 30 Hz: external viewer refresh.

ENCODER_COUNTS = 4096


REFERENCE_SPEED = 0.6

# Distance along the original route from frozen progress to the single rejoin target
# Saved submission trials exercise this single target on two cube layouts.
DETOUR_REJOIN_DISTANCE = 2.0

# None gives a fresh obstacle layout each run and an integer repeats the same samples
OBSTACLE_SEED = 42


OMEGA_MAX = 2


INITIAL_POSE = np.array([0.3, -2.0, np.deg2rad(10.0)])


INITIAL_COVARIANCE = np.diag([
    0.3**2,
    0.3**2,
    np.deg2rad(10.0)**2,
])

def main(show_viewer=True, max_time=None, camera_diagnostics=False,
         estimator_diagnostics=True, prediction_log=None, reports=None):

    robot, path, controller, estimator, logger = setup(
        show_viewer, camera_diagnostics, estimator_diagnostics, prediction_log, reports)
    docking = DockingSession(robot)
    navigation = Navigation()
    command = (0.0, 0.0)
    collision = None
    pose, reference = None, None
    wall_start = time.perf_counter()
    if max_time is None:
        max_time = 5 * path.total_length / REFERENCE_SPEED + 20

    try:
        while robot.running() and robot.time < max_time:
            updates = updates_due(robot.time)

            # 1. Read the latest sampled wheel encoders.
            encoders = robot.read_encoders()
            if updates.lidar:
                scan = robot.read_lidar()

            # 2. Predict from wheel motion, then correct using a NEW camera image.
            if updates.ekf and estimator is not None:
                estimator.predict(encoders)

            if updates.camera and estimator is not None:
                measurement = read_localisation_camera(robot, show_viewer)
                logger.record_camera_update(robot, estimator, measurement)
                estimator.correct(measurement)

            # Pair each instantaneous scan with the EKF pose at the same simulation time
            # Copy the pose after prediction and camera correction so later updates cannot move it
            if updates.lidar:
                scan_pose = np.array(estimator.get_estimate()["pose"], dtype=float)

            # 3. Update navigation and the body-speed command.
            if updates.control:
                pose = estimator.get_estimate()["pose"]

                # Map once after stopping then try a detour to the single original route target
                # Use the existing mounting offset shared by collision checking and plotting
                if updates.lidar and navigation.state == "BLOCKED":
                    navigation.update_obstacle_map(
                        path.grid, scan, scan_pose, collision["lidar_offset"])
                    navigation.plan_detour(path, pose, DETOUR_REJOIN_DISTANCE)

                # Navigation advances only the active path and restores saved progress at rejoin
                # LQR still owns the calculation of the proposed body command
                reference = navigation.update_reference(path, pose)

                if reference["stop"]:
                    proposed_command = docking.update(robot, encoders)
                else:
                    proposed_command = controller.update(pose, reference, CONTROL_DT)

                # Limit the proposal without running PI or applying new torque yet
                proposed_command = robot.set_body_command(proposed_command)

                # Check the active path proposal even while the executed command is zero
                # At rejoin check the new original route command immediately rather than reusing a detour result
                if updates.lidar or navigation.rejoined:
                    collision = check_collision(
                        scan, proposed_command, scan_pose=scan_pose, current_pose=pose)

                command = proposed_command
                if not reference["stop"]:
                    speed, yaw_rate = encoder_body_speed(robot, encoders)
                    command = navigation.update(
                        proposed_command, collision, speed, yaw_rate, updates.lidar)

            # 4. Convert the body command to wheel targets and run wheel PI.
            limited_command = robot.set_body_command(command)
            robot.update_wheel_pi(encoders)  # PI_DT timing and torque hold live in Robot.

            # Display the proposed path so a blocked route stays visible while stopped
            if updates.lidar:
                logger.display_lidar(scan, collision)

            # Refresh the map after planning and clear the detour overlay immediately after rejoining
            # Drawing stays in evaluation so it cannot change navigation or controller decisions
            if updates.lidar or (updates.control and navigation.rejoined):
                logger.display_plan(path, navigation.detour, navigation.state)

            # Supporting recording/display stays below the core sensing/control steps.
            logger.record_step(robot, updates, estimator, pose, reference,
                               command, limited_command, docking)
            robot.step()
            logger.record_contacts(robot)
            if updates.viewer and show_viewer:
                update_viewer(robot, logger, wall_start)

            if docking.stop_time is not None and robot.time - docking.stop_time >= 2.0:
                break

        log = logger.finish(robot, path, docking.controller, docking.start_time,
                            docking.stop_time, docking.camera)
    finally:
        robot.close()
        logger.close_display()

    logger.save_reports(show=show_viewer)
    return log


# Supporting setup, timing, camera experiment and docking details.

def encoder_body_speed(robot, encoders):
    # Convert measured wheel speeds into forward speed and turn rate
    speed = robot.wheel_radius * (encoders['left_speed'] + encoders['right_speed']) / 2
    yaw_rate = robot.wheel_radius * (encoders['right_speed'] - encoders['left_speed']) / robot.wheel_separation
    return speed, yaw_rate


def setup(show_viewer, camera_diagnostics, estimator_diagnostics, prediction_log, reports):
    scene = Path(__file__).resolve().parent.parent / "robotis_tb3" / "scene_loop.xml"
    grid, start = map_from_scene(scene)
    path = PlannedLoopReference(grid, start, REFERENCE_SPEED)
    print(f"Planned {path.total_length:.1f} m loop from the known wall map.")
    controller = Controller(REFERENCE_SPEED, CONTROL_DT, verbose=LOG_RESULTS)

    # Write the sampled cube sizes and x positions before MuJoCo loads the scene
    # The wall map excludes these cubes and navigation still has to detect them with LiDAR
    obstacle_layout = randomise_room_obstacles(scene, path, seed=OBSTACLE_SEED)

    robot = Robot(scene, max_yaw_rate=OMEGA_MAX, show_viewer=show_viewer,
                  physics_dt=PHYSICS_DT, encoder_dt=ENCODER_DT, pi_dt=PI_DT,
                  encoder_speed_window=ENCODER_SPEED_WINDOW, encoder_counts=ENCODER_COUNTS)
    robot.reset((*path.start_position, path.start_heading))

    settings = dict(
        physics_dt=robot.physics_dt, encoder_dt=ENCODER_DT,
        encoder_speed_window=ENCODER_SPEED_WINDOW, pi_dt=PI_DT,
        control_dt=CONTROL_DT, ekf_dt=EKF_DT, camera_dt=CAMERA_DT, lidar_dt=LIDAR_DT,
        docking_dt=DOCKING_DT, encoder_counts=ENCODER_COUNTS, scene=str(scene),
        obstacle_seed=OBSTACLE_SEED, obstacle_layout=obstacle_layout)
    logger = RunLogger(robot, path, controller, settings,
                       controller_enabled=LOG_RESULTS, estimator_enabled=estimator_diagnostics,
                       prediction_log=prediction_log, reports=reports,
                       show_viewer=show_viewer, camera_diagnostics=camera_diagnostics)
    # Feedback always needs the EKF, even when its diagnostic logging is disabled.
    estimator = PoseEKF(INITIAL_POSE, INITIAL_COVARIANCE, {
        "wheel_radius": robot.wheel_radius, "wheel_base": robot.wheel_separation})
    logger.configure_estimator(
        estimator, CAMERA_DROPOUT_PERIODS if CAMERA_DROPOUTS else [])
    if show_viewer:
        logger.open_display(path)
    return robot, path, controller, estimator, logger


def updates_due(simulation_time):
    # Sensor/control intervals use whole physics steps; the viewer interval is
    # rounded to the nearest step. Camera and EKF events therefore stay aligned.
    step = round(simulation_time / PHYSICS_DT)
    return SimpleNamespace(**{
        name: step % round(interval / PHYSICS_DT) == 0
        for name, interval in (
            ("ekf", EKF_DT), ("camera", CAMERA_DT), ("lidar", LIDAR_DT), ("control", CONTROL_DT),
            ("display", CAMERA_DISPLAY_DT), ("viewer", VIEWER_DT))})


def read_localisation_camera(robot, annotate=False):
    camera = robot.read_camera()
    if CAMERA_DROPOUTS and any(start <= robot.time < end
                              for start, end in CAMERA_DROPOUT_PERIODS):
        camera["rgb"][:] = 0
    return robot.read_camera_pose(camera, annotate=annotate)


def update_viewer(robot, logger, wall_start):
    robot.sync_viewer()
    logger.flush_display()
    # Pace interactive runs to real time; headless runs never wait here.
    remaining = wall_start + robot.time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


class DockingSession:
    """Keep docking progress and its camera timing out of the navigation loop."""

    def __init__(self, robot):
        self.controller = VisualDocking()
        self.detector = StationDetector(robot.model.camera('robot_camera').fovy[0])
        self.start_time = None
        self.stop_time = None
        self.next_update = 0.0
        self.camera = None
        self.command = (0.0, 0.0)

    def update(self, robot, encoders):
        if self.start_time is None:
            self.start_time = robot.time
            print('Lap complete. Acquiring charging-station marker for visual docking.')
        if self.stop_time is not None:
            return 0.0, 0.0
        if robot.time - self.start_time > 60:
            self.stop_time = robot.time
            print('Docking timed out; stopping without claiming success.')
            return 0.0, 0.0
        if robot.time + 1e-10 < self.next_update:
            return self.command

        self.camera = robot.read_camera()
        observation = self.detector.observe(self.camera['rgb'], self.camera['timestamp'])
        speed, yaw = encoder_body_speed(robot, encoders)
        self.command = self.controller.update(observation, speed, yaw)
        self.next_update = robot.time + DOCKING_DT
        if self.controller.finished:
            self.stop_time = robot.time
            print('Docking position reached (camera-based stop; no contact sensor).')
        return self.command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without viewer or pacing.")
    parser.add_argument("--max-time", type=float, help="Maximum simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=OBSTACLE_SEED, help="Repeatable room-cube layout.")
    parser.add_argument("--camera-dropouts", action="store_true", help="Use the documented camera outage periods.")
    parser.add_argument("--output", type=Path, help="Report folder; use a different folder for each trial.")
    parser.add_argument("--plots", nargs="*", choices=("estimator", "controller", "body", "pi"),
                        default=None, help="Override evaluation.py reports; no names disables reports.")
    args = parser.parse_args()
    OBSTACLE_SEED = args.seed
    CAMERA_DROPOUTS = args.camera_dropouts
    if args.output is not None:
        import evaluation
        evaluation.FULL_SYSTEM_FOLDER = args.output
    main(show_viewer=not args.headless, max_time=args.max_time, reports=args.plots)

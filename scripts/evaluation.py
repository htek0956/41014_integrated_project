"""Run recording and diagnostic displays; truth is used for evaluation only."""

from pathlib import Path

import numpy as np

from controller import tracking_errors
from models import wrap


# Choose any of: "estimator", "controller", "body", "pi". Use () for no reports.
# Controller, body and PI reports share the same recorded data and metrics.
REPORTS = ("estimator", "controller", "body", "pi")
SHOW_REPORTS = False             # Headless runs save without opening windows.
FULL_SYSTEM_FOLDER = Path(__file__).resolve().parents[1] / "figures" / "full_system"


def camera_innovation_nis(estimator, measurement):
    """Calculate innovation consistency before correction changes the filter."""
    if not measurement["valid"]:
        return np.nan
    residual = np.asarray(measurement["pose"]) - estimator.state
    residual[2] = wrap(residual[2])
    S = estimator.covariance + measurement["covariance"]
    return float(residual @ np.linalg.solve(S, residual))


class RunLogger:
    def __init__(self, robot, path, controller, settings, *,
                 controller_enabled=False, estimator_enabled=True,
                 prediction_log=None, reports=None, show_viewer=False,
                 camera_diagnostics=False):
        self.show_viewer = show_viewer
        self.camera_diagnostics = camera_diagnostics
        self.camera_measurement = None
        self.camera_truth = None
        self.nis = np.nan
        self.reports = tuple(REPORTS if reports is None else reports)
        unknown = set(self.reports) - {"estimator", "controller", "body", "pi"}
        if unknown:
            raise ValueError(f"Unknown reports: {sorted(unknown)}")
        self.prediction = prediction_log
        self.contact_steps = 0
        self.unwanted_contact_steps = 0
        self.station_contact_steps = 0
        self.contact_geoms = set()
        base = robot.model.body("base").id
        self.robot_geoms = set()
        for geom in range(robot.model.ngeom):
            body = int(robot.model.geom_bodyid[geom])
            while body != 0 and body != base:
                body = int(robot.model.body_parentid[body])
            if body == base:
                self.robot_geoms.add(geom)
        self.obstacle_ids = {
            i for i in range(robot.model.ngeom)
            if (robot.model.geom(i).name or '').startswith(
                ('nav_wall_', 'scenery_wall_', 'dock_', 'obstacle_'))}
        self.camera_samples = []
        self.plan_figure = None
        self.lidar_plot = None
        self.estimator = None
        self.control = None
        gains = [{"kp": c.kp, "ki": c.ki} for c in robot.wheel_controllers]

        if controller_enabled or set(self.reports) & {"controller", "body", "pi"}:
            self.control = {name: [] for name in (
                "time", "pose", "reference", "cross_track_error", "heading_error",
                "command", "limited_command")}
            self.control["wheel"] = {name: [] for name in (
                "time", "target", "measured", "torque", "raw_torque",
                "body_command", "limited_command", "body_measured", "pi_updated")}
            self.control["reference_path"] = np.array([
                [point["x"], point["y"]]
                for point in (path.reference_at(s)
                              for s in np.linspace(0, path.total_length, 1000))])
            self.control["settings"] = {
                **settings, "reference_speed": path.speed, "path": "map_planned_loop",
                "lqr_gain": controller.K_lqr.tolist(), "pi_gains": gains,
                "torque_limits": robot.model.actuator_ctrlrange[robot.actuators].tolist(),
                "max_wheel_speed": robot.max_wheel_speed,
                "max_yaw_rate": robot.max_yaw_rate}

        if estimator_enabled:
            self.estimator = {name: [] for name in (
                "time", "truth", "estimate", "covariance", "camera_updated",
                "camera_valid", "camera_pose", "tag_count", "reprojection_rms_px", "nis")}
            self.estimator["settings"] = dict(settings)

        if self.prediction is not None:
            self.prediction.update({name: [] for name in (
                "time", "truth", "encoder_angles", "ideal_encoder_angles",
                "body_speed", "phase", "contact_steps")})
            self.prediction["settings"] = {
                **settings, "wheel_radius": robot.wheel_radius,
                "wheel_base": robot.wheel_separation,
                "reference_speed": path.speed, "pi_gains": gains}

    def record_camera_update(self, robot, estimator, measurement):
        # Retain same-time truth for evaluation, never for EKF correction.
        self.camera_measurement = measurement
        self.camera_truth = robot.get_truth()
        self.nis = camera_innovation_nis(estimator, measurement)

    def record_step(self, robot, updates, estimator, pose, reference,
                    command, limited_command, docking):
        """Keep recording schedules and display details outside the core loop."""
        if updates.ekf:
            self.record_prediction(robot, docking.start_time is not None)
            if estimator is not None:
                self.record_estimator(robot, estimator.get_estimate(),
                                      self.camera_measurement, updates.camera,
                                      self.nis if updates.camera else np.nan)
        if updates.control:
            self.record_control(robot, pose, reference, command, limited_command)
        self.record_wheels(robot, command, limited_command)

        if updates.display and robot.time > 0 and (self.show_viewer or self.camera_diagnostics):
            if estimator is None:
                measurement = robot.read_camera_pose(annotate=self.show_viewer)
                truth = robot.get_truth()
            else:
                measurement, truth = self.camera_measurement, self.camera_truth
            self.record_camera(measurement, truth, show=self.show_viewer)

    def configure_estimator(self, estimator, dropout_periods):
        if self.estimator is not None:
            self.estimator["settings"].update(
                initial_pose=estimator.state.tolist(),
                initial_covariance=estimator.covariance.tolist(),
                process_covariance_per_prediction=estimator.Q.tolist(),
                feedback="EKF estimated pose",
                camera_covariance="Approximate PnP covariance; not calibrated",
                camera_dropout_periods_s=dropout_periods)

    def record_prediction(self, robot, docking_started):
        if self.prediction is None:
            return
        enc, ideal = robot.read_encoders(), robot.read_encoder_truth()
        sample = dict(
            time=robot.time, truth=robot.get_truth(),
            encoder_angles=[enc["left_angle"], enc["right_angle"]],
            ideal_encoder_angles=[ideal["left_angle"], ideal["right_angle"]],
            body_speed=robot.get_truth_velocity(),
            phase="docking" if docking_started else "navigation",
            contact_steps=self.contact_steps)
        for name, value in sample.items():
            self.prediction[name].append(value)

    def record_estimator(self, robot, estimate, measurement, camera_updated, nis):
        if self.estimator is None:
            return
        valid = measurement["valid"]
        sample = dict(
            time=robot.time, truth=robot.get_truth(), estimate=estimate["pose"],
            covariance=estimate["covariance"].copy(), camera_updated=camera_updated,
            # Availability is held between images; camera_updated marks new frames.
            camera_valid=valid, camera_pose=measurement["pose"].copy()
            if camera_updated and valid else np.full(3, np.nan),
            tag_count=len(measurement["ids_used"]), nis=nis,
            reprojection_rms_px=measurement["reprojection_rms_px"]
            if camera_updated and valid else np.nan)
        for name, value in sample.items():
            self.estimator[name].append(value)

    def record_control(self, robot, pose, reference, command, limited_command):
        if self.control is None:
            return
        cross_track, heading = tracking_errors(pose, reference)
        sample = dict(time=robot.time, pose=tuple(pose), reference=reference.copy(),
                      cross_track_error=cross_track, heading_error=heading,
                      command=command, limited_command=limited_command)
        for name, value in sample.items():
            self.control[name].append(value)

    def record_wheels(self, robot, command, limited_command):
        if self.control is None:
            return
        sample = dict(
            time=robot.time, target=robot.wheel_targets.copy(),
            measured=robot.pi_feedback_speed.copy(), pi_updated=robot.pi_updated,
            torque=robot.wheel_torques.copy(),
            raw_torque=[wheel.raw_torque for wheel in robot.wheel_controllers],
            body_command=command, limited_command=limited_command,
            body_measured=robot.get_truth_velocity())
        for name, value in sample.items():
            self.control["wheel"][name].append(value)

    def record_contacts(self, robot):
        # Count physics steps with robot contact, not contacts among scenery.
        names = set()
        for contact in robot.data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.robot_geoms and second in self.obstacle_ids:
                names.add(robot.model.geom(second).name)
            if second in self.robot_geoms and first in self.obstacle_ids:
                names.add(robot.model.geom(first).name)
        self.contact_geoms.update(names)
        self.contact_steps += int(bool(names))
        self.station_contact_steps += int(any(name.startswith('dock_') for name in names))
        self.unwanted_contact_steps += int(any(not name.startswith('dock_') for name in names))

    def open_display(self, path):
        import matplotlib.pyplot as plt
        from path_planning import plot_plan
        from plot_lidar import LidarPlot

        self.lidar_plot = LidarPlot()
        self.plan_figure = plot_plan(path)
        axis = self.plan_figure.axes[0]

        # Keep the original blue route and overlay the temporary detour in world coordinates
        # Empty lines can be updated without rebuilding the figure or adding repeated legend entries
        self.plan_map = axis.images[0]
        self.detour_line, = axis.plot([], [], color='tab:orange', linewidth=2.5,
                                     label='LiDAR detour')
        self.rejoin_marker, = axis.plot([], [], 'x', color='tab:orange', markersize=9,
                                       label='Rejoin point')

        self.truth_marker, = axis.plot([], [], '+', color='tab:orange', markersize=12,
                                       label='Truth (verification only)')
        self.camera_marker, = axis.plot([], [], 'o', color='tab:green', markersize=6,
                                        label='Camera-only pose')
        axis.legend()
        self.plan_figure.canvas.manager.set_window_title('Map and planned route')
        plt.show(block=False)
        self.plan_figure.canvas.draw()
        self.flush_display()

    def display_plan(self, path, detour, state):
        """Refresh the live map with detected cells and the current detour"""

        # Headless runs have no figure and a closed window should stay closed
        if self.plan_figure is None:
            return

        import matplotlib.pyplot as plt
        if not plt.fignum_exists(self.plan_figure.number):
            return

        # Map bounds stay fixed while LiDAR adds permanently occupied cells
        self.plan_map.set_data(path.grid.occupied)

        if detour is not None:
            # Display the proposed detour while waiting for clear scans as well as while following it
            self.detour_line.set_data(detour.sample_xy[:, 0], detour.sample_xy[:, 1])
            self.rejoin_marker.set_data([detour.end_position[0]], [detour.end_position[1]])
        else:
            # Remove an old detour after rejoining or discarding it for a new blockage
            self.detour_line.set_data([], [])
            self.rejoin_marker.set_data([], [])

        self.plan_figure.axes[0].set_title(f'Map and planned route | {state}')
        self.plan_figure.canvas.draw_idle()

    def display_lidar(self, scan, collision):
        if self.lidar_plot is not None:
            self.lidar_plot.update(scan, collision)

    def record_camera(self, measurement, truth, show=False):
        import cv2

        truth = np.asarray(truth)
        valid = measurement["valid"]
        self.camera_samples.append(dict(
            timestamp=measurement["timestamp"],
            pose=measurement["pose"].copy() if valid else None,
            truth=truth.copy(), ids_used=list(measurement["ids_used"]),
            covariance=measurement["covariance"]))
        if not show:
            return
        rgb = measurement["annotated_rgb"].copy()
        self.truth_marker.set_data([truth[0]], [truth[1]])
        if valid:
            pose = measurement["pose"]
            position_error = np.linalg.norm(pose[:2] - truth[:2])
            heading_error = wrap(pose[2] - truth[2])
            comparison = (f'TRUTH CHECK ONLY: error {position_error*100:.1f}cm / '
                          f'{np.rad2deg(heading_error):+.2f}deg')
            self.camera_marker.set_data([pose[0]], [pose[1]])
        else:
            comparison = 'No camera measurement; no stale pose returned'
            self.camera_marker.set_data([], [])
        cv2.rectangle(rgb, (0, rgb.shape[0]-32), (rgb.shape[1], rgb.shape[0]), (20, 20, 20), -1)
        cv2.putText(rgb, comparison, (8, rgb.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (255, 255, 255), 1, cv2.LINE_AA)
        import matplotlib.pyplot as plt
        if plt.fignum_exists(self.plan_figure.number):
            self.plan_figure.canvas.draw_idle()
        cv2.imshow('Robot camera', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def flush_display(self):
        if self.plan_figure is not None:
            import matplotlib.pyplot as plt
            if plt.fignum_exists(self.plan_figure.number):
                self.plan_figure.canvas.flush_events()

    def finish(self, robot, path, docking, docking_start, stop_time, latest_camera):
        final_pose = robot.get_truth()
        dock_target = robot.model.site('dock_target').pos[:2]
        endpoint_error = float(np.linalg.norm(np.asarray(final_pose[:2])-dock_target))
        summary = dict(lap_completed=bool(path.finished), docked=bool(docking.finished),
                       duration_s=robot.time, obstacle_contact_steps=self.contact_steps,
                       unwanted_contact_steps=self.unwanted_contact_steps,
                       station_contact_steps=self.station_contact_steps,
                       contact_geoms=sorted(self.contact_geoms))
        print(f'Lap completed: {path.finished}; docked: {docking.finished}; '
              f'time: {robot.time:.2f} s; progress: {path.progress:.2f}/{path.total_length:.2f} m; '
              f'dock error: {endpoint_error:.3f} m; obstacle-contact steps: {self.contact_steps}')
        if self.prediction is not None:
            self.prediction["settings"].update(summary)
        if self.estimator is not None:
            self.estimator["settings"].update(summary)
        if self.control is not None:
            self.control["settings"].update(summary)
            self.control.update(
                completed=docking.finished, lap_completed=path.finished,
                docked=docking.finished, dock_mode=docking.mode,
                obstacle_contact_steps=self.contact_steps,
                final_body_velocity=robot.get_truth_velocity(),
                docking_duration_s=None if docking_start is None else robot.time-docking_start,
                docking_camera_rgb=None if latest_camera is None else latest_camera['rgb'].copy(),
                camera_measurements=self.camera_samples, final_pose=final_pose,
                endpoint_error=endpoint_error, stop_time=stop_time, duration=robot.time)
            if self.estimator is not None:
                self.control["estimator"] = self.estimator
            if self.control["cross_track_error"]:
                rms = np.sqrt(np.mean(np.square(self.control["cross_track_error"])))
                print(f'Reference cross-track RMS, including deliberate docking departure: {rms:.3f} m')
        return self.control

    def save_reports(self, show=False):
        # Existing plotting scripts own their figures, metrics and numeric exports.
        display = show and SHOW_REPORTS
        if "estimator" in self.reports and self.estimator is not None and self.estimator["time"]:
            from plot_estimator import save_estimator_results
            save_estimator_results(self.estimator, show=False, close=not display,
                                   output_folder=FULL_SYSTEM_FOLDER)
        selected = tuple(name for name in self.reports if name != "estimator")
        if selected and self.control is not None and self.control["time"]:
            from plot_controller_results import save_controller_results
            save_controller_results(self.control, show=False, close=not display,
                                    reports=selected, output_folder=FULL_SYSTEM_FOLDER)
        # Opening windows blocks execution: save EVERY report before displaying.
        if display:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close('all')

    def close_display(self):
        import cv2
        cv2.destroyAllWindows()
        if self.lidar_plot is not None:
            self.lidar_plot.close()
        if self.plan_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.plan_figure)

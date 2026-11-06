import math
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from wheel_controller import WheelPI
from models import encoder_angle_measurement

LIDAR_BEAMS = 360
LIDAR_MIN_RANGE = 0.12
LIDAR_MAX_RANGE = 3.5
# LDS-01 specifications: https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_01/


class Robot:


    def __init__(self, scene_path, max_yaw_rate, show_viewer=True,
                 physics_dt=None, encoder_dt=None, pi_dt=0.01,
                 encoder_speed_window=0.02, encoder_counts=4096):

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        if physics_dt is not None:
            if not np.isfinite(physics_dt) or physics_dt <= 0:
                raise ValueError("Physics interval must be finite and positive.")
            self.model.opt.timestep = physics_dt

        self.encoder_dt = self.physics_dt if encoder_dt is None else encoder_dt
        self.pi_dt = pi_dt
        self.encoder_speed_window = encoder_speed_window
        self.encoder_counts = encoder_counts

        # Align sensor and PI events with physics steps; the speed window
        # contains complete encoder intervals. Eleven samples span 20 ms at 2 ms.
        for name, interval, base_dt in (
            ("Encoder interval", self.encoder_dt, self.physics_dt),
            ("PI interval", self.pi_dt, self.physics_dt),
            ("Speed window", self.encoder_speed_window, self.encoder_dt),
        ):
            if (not np.isfinite(interval) or interval < base_dt
                    or not np.isclose(interval / base_dt, round(interval / base_dt), atol=1e-9, rtol=0)):
                raise ValueError(f"{name} must be a positive whole multiple of {base_dt} s.")
        encoder_angle_measurement(0.0, self.encoder_counts)
        window_intervals = round(self.encoder_speed_window / self.encoder_dt)
        self.encoder_history = deque(maxlen=window_intervals + 1)

        self.data = mujoco.MjData(self.model)

        self.lidar_site = self.model.site("lidar").id
        self.lidar_body = self.model.body("base").id
        self.lidar_angles = np.linspace(-np.pi, np.pi, LIDAR_BEAMS, endpoint=False)
        # Sensor frame: +X forward, +Y left, +Z up. All beams lie in its XY plane.
        self.lidar_directions = np.column_stack((
            np.cos(self.lidar_angles), np.sin(self.lidar_angles), np.zeros(LIDAR_BEAMS)))
        self.lidar_geom_ids = np.empty(LIDAR_BEAMS, dtype=np.int32)
        self.lidar_ranges = np.empty(LIDAR_BEAMS)
        self.lidar_rng = np.random.default_rng(42)
        self.lidar_noise_std = 0.02  # metres; assumed Gaussian noise, could not find a datasheet sigma.
        self.landmark_map_path = Path(scene_path).with_suffix('.landmarks.json')
        self._camera_pose_estimator = None

        self.renderer = mujoco.Renderer(
            self.model,
            height=480,
            width=640,
        )

        wheel_names = ("wheel_left", "wheel_right")


        self.actuators = [
            self.model.actuator(name).id for name in wheel_names
        ]
        self.position_indices = [
            int(self.model.joint(name).qposadr[0]) for name in wheel_names
        ]
        self.velocity_indices = [
            int(self.model.joint(name).dofadr[0]) for name in wheel_names
        ]
        self.base_position_index = int(
            self.model.joint("base_joint").qposadr[0]
        )


        left_body = self.model.body("wheel_left").id
        right_body = self.model.body("wheel_right").id

        if self.model.body_parentid[left_body] != self.model.body_parentid[right_body]:
            raise ValueError("Wheel separation requires a shared wheel-body parent.")

        self.wheel_separation = float(np.linalg.norm(
            self.model.body_pos[left_body] - self.model.body_pos[right_body]
        ))


        self.wheel_radius = 0.033


        self.max_wheel_speed = 25
        self.max_yaw_rate = max_yaw_rate


        self.wheel_controllers = [
            WheelPI(self.model.actuator_ctrlrange[actuator])
            for actuator in self.actuators
        ]

        self.wheel_targets = np.zeros(2)
        self.wheel_torques = np.zeros(2)
        self.pi_feedback_speed = np.zeros(2)

        self.viewer = None
        self.closed = False
        self.reset()


        if show_viewer:

            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False,
            )


            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = self.model.body("base").id
                self.viewer.cam.distance = 2.5
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -30

    @property
    def time(self):

        return float(self.data.time)

    @property
    def physics_dt(self):

        return float(self.model.opt.timestep)

    def reset(self, pose=None):

        mujoco.mj_resetData(self.model, self.data)

        if pose is not None:
            x, y, heading = pose
            start = self.base_position_index
            self.data.qpos[start:start + 2] = (x, y)


            self.data.qpos[start + 3:start + 7] = (
                math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)
            )

        for controller in self.wheel_controllers:
            controller.reset()

        self.wheel_targets[:] = 0.0
        self.wheel_torques[:] = 0.0
        self.pi_feedback_speed[:] = 0.0
        self.encoder_history.clear()
        self.next_encoder_time = 0.0
        self.next_pi_time = 0.0
        self.previous_pi_time = None
        self.pi_updated = False


        mujoco.mj_forward(self.model, self.data)
        self._sample_encoders()

    def running(self):

        return not self.closed and (
            self.viewer is None or self.viewer.is_running()
        )

    def read_encoders(self):

        # Return a copy of the latest sample. Multiple consumers at the same
        # time must not append duplicate readings or change sensor history.
        return self.encoder_measurement.copy()

    def _sample_encoders(self):

        if self.time + 1e-10 < self.next_encoder_time:
            return

        angles = encoder_angle_measurement(
            self.data.qpos[self.position_indices], self.encoder_counts)
        self.encoder_history.append((self.time, float(angles[0]), float(angles[1])))

        oldest_time, oldest_left, oldest_right = self.encoder_history[0]
        elapsed = self.time - oldest_time
        speeds = np.zeros(2)
        if elapsed > 0:
            speeds = (angles - np.array([oldest_left, oldest_right])) / elapsed

        self.encoder_measurement = {
            "timestamp": self.time,
            "left_angle": float(angles[0]),
            "right_angle": float(angles[1]),
            "left_speed": float(speeds[0]),
            "right_speed": float(speeds[1]),
        }
        self.next_encoder_time += self.encoder_dt

    def read_encoder_truth(self):

        # Ideal joint values for labelled diagnostics only. Operational PI and
        # EKF feedback must use read_encoders(), which includes quantisation.
        left_position, right_position = self.position_indices
        left_velocity, right_velocity = self.velocity_indices

        return {
            "timestamp": self.time,
            "left_angle": float(self.data.qpos[left_position]),
            "right_angle": float(self.data.qpos[right_position]),
            "left_speed": float(self.data.qvel[left_velocity]),
            "right_speed": float(self.data.qvel[right_velocity]),
        }

    def read_camera(self):


        self.renderer.update_scene(self.data, camera="robot_camera")
        rgb = self.renderer.render().copy()

        return {"timestamp": self.time, "rgb": rgb}

    def read_lidar(self):

        # Cast one instantaneous scan from the current sensor mounting pose.
        origin = self.data.site_xpos[self.lidar_site]
        rotation = self.data.site_xmat[self.lidar_site].reshape(3, 3)
        directions_world = self.lidar_directions @ rotation.T
        mujoco.mj_multiRay(
            self.model, self.data, origin, directions_world.ravel(),
            None, True, self.lidar_body, self.lidar_geom_ids, self.lidar_ranges,
            None, LIDAR_BEAMS, LIDAR_MAX_RANGE)

        # Excluding base removes the chassis/scanner itself. The wheel bodies
        # are below the scan plane. Include static walls and moving obstacles.
        ranges = self.lidar_ranges.copy()
        ranges[(ranges < 0) | (ranges > LIDAR_MAX_RANGE)] = np.inf
        ranges[ranges < LIDAR_MIN_RANGE] = np.nan

        # Noise affects valid returns only; missing/out-of-range beams stay unchanged.
        valid = np.isfinite(ranges)
        ranges[valid] += self.lidar_rng.normal(0.0, self.lidar_noise_std, valid.sum())
        ranges[valid & ((ranges < LIDAR_MIN_RANGE) | (ranges > LIDAR_MAX_RANGE))] = np.nan

        return dict(timestamp=self.time, angles=self.lidar_angles.copy(), ranges=ranges,
                    range_min=LIDAR_MIN_RANGE, range_max=LIDAR_MAX_RANGE, frame="lidar")

    def read_camera_pose(self, camera_measurement=None, annotate=False):

        from camera_localisation import CameraPoseEstimator, transform

        frame = self.read_camera() if camera_measurement is None else camera_measurement
        if self._camera_pose_estimator is None:
            camera = self.model.camera('robot_camera')
            if camera.bodyid[0] != self.model.body('base').id:
                raise ValueError('Camera extrinsics require a fixed camera on base.')
            height, width = frame['rgb'].shape[:2]
            focal = height/(2*np.tan(np.deg2rad(camera.fovy[0])/2))
            intrinsics = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, camera.quat)


            base_from_optical = transform(rotation.reshape(3, 3) @ np.diag([1., -1., -1.]), camera.pos)
            self._camera_pose_estimator = CameraPoseEstimator(self.landmark_map_path, intrinsics, base_from_optical)
        measurement = self._camera_pose_estimator.estimate(frame['rgb'], frame['timestamp'])
        if annotate:
            measurement['annotated_rgb'] = self._camera_pose_estimator.annotate(frame['rgb'], measurement)
        return measurement

    def apply_body_command(self, command, dt=None):

        # Convenience wrapper for the existing diagnostic scripts.
        if dt is not None and (not np.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive.")
        limited_command = self.set_body_command(command)
        self.update_wheel_pi(self.read_encoders())
        return limited_command

    def set_body_command(self, command):
        """Convert body speed/yaw rate to limited wheel-speed targets."""
        forward_speed, yaw_rate = command

        if not np.all(np.isfinite([forward_speed, yaw_rate])):
            raise ValueError("Command must be finite.")
        yaw_rate = float(np.clip(
            yaw_rate, -self.max_yaw_rate, self.max_yaw_rate
        ))


        half_separation = self.wheel_separation / 2
        targets = np.array([
            (forward_speed - half_separation * yaw_rate) / self.wheel_radius,
            (forward_speed + half_separation * yaw_rate) / self.wheel_radius,
        ])


        peak_speed = np.max(np.abs(targets))
        if peak_speed > self.max_wheel_speed:
            targets *= self.max_wheel_speed / peak_speed

        self.wheel_targets[:] = targets

        limited_speed = self.wheel_radius * (targets[0] + targets[1]) / 2
        limited_yaw_rate = (
            self.wheel_radius * (targets[1] - targets[0])
            / self.wheel_separation
        )
        return float(limited_speed), float(limited_yaw_rate)

    def update_wheel_pi(self, encoders):
        """Run PI when PI_DT is due; hold the last torque between updates."""
        self.pi_updated = self.time + 1e-10 >= self.next_pi_time
        if self.pi_updated:
            self.pi_feedback_speed[:] = [encoders["left_speed"], encoders["right_speed"]]
            pi_interval = (self.pi_dt if self.previous_pi_time is None
                           else self.time - self.previous_pi_time)
            for wheel, controller in enumerate(self.wheel_controllers):
                self.wheel_torques[wheel] = controller.update(
                    self.wheel_targets[wheel], self.pi_feedback_speed[wheel], pi_interval)

            self.previous_pi_time = self.time
            self.next_pi_time = self.time + self.pi_dt


        self.data.ctrl[self.actuators] = self.wheel_torques

    def step(self):

        mujoco.mj_step(self.model, self.data)
        self._sample_encoders()

    def sync_viewer(self):

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_truth(self):

        start = self.base_position_index
        x, y = self.data.qpos[start:start + 2]
        qw, qx, qy, qz = self.data.qpos[start + 3:start + 7]

        heading = math.atan2(
            2 * (qw * qz + qx * qy),
            1 - 2 * (qy**2 + qz**2),
        )
        return float(x), float(y), heading

    def get_truth_velocity(self):

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.model.body("base").id, velocity, 0,
        )
        heading = self.get_truth()[2]
        forward_speed = (
            velocity[3] * math.cos(heading) + velocity[4] * math.sin(heading)
        )
        return float(forward_speed), float(velocity[2])

    def close(self):

        self.data.ctrl[self.actuators] = 0.0

        if self.viewer is not None:
            self.viewer.close()

        self.renderer.close()
        self.closed = True

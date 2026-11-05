import math
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from wheel_controller import WheelPI


class Robot:


    def __init__(self, scene_path, max_yaw_rate, show_viewer=True):

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
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


        mujoco.mj_forward(self.model, self.data)

    def running(self):

        return not self.closed and (
            self.viewer is None or self.viewer.is_running()
        )

    def read_encoders(self):

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

        raise NotImplementedError("Read a timestamped LiDAR scan.")

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

    def apply_body_command(self, command, dt):

        forward_speed, yaw_rate = command

        if dt <= 0 or not np.all(np.isfinite([forward_speed, yaw_rate, dt])):
            raise ValueError("Command and dt must be finite, with dt positive.")

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

        for wheel, (controller, velocity_index) in enumerate(zip(
            self.wheel_controllers, self.velocity_indices
        )):
            measured_speed = float(self.data.qvel[velocity_index])
            self.wheel_torques[wheel] = controller.update(
                targets[wheel], measured_speed, dt
            )


        self.data.ctrl[self.actuators] = self.wheel_torques

        limited_speed = self.wheel_radius * (targets[0] + targets[1]) / 2
        limited_yaw_rate = (
            self.wheel_radius * (targets[1] - targets[0])
            / self.wheel_separation
        )
        return float(limited_speed), float(limited_yaw_rate)

    def step(self):

        mujoco.mj_step(self.model, self.data)

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

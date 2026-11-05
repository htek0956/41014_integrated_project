"""MuJoCo sensor/actuator interface, reusing the wheel-control experiments."""

import math

import mujoco
import mujoco.viewer
import numpy as np

from wheel_controller import WheelPI


class Robot:
    """Own simulation resources and one PI controller per drive wheel."""

    def __init__(self, scene_path, max_yaw_rate, show_viewer=True):
        
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(
            self.model,
            height=480,
            width=640,
        )
        
        wheel_names = ("wheel_left", "wheel_right")

        # Free-joint position and velocity arrays use different indexing.
        # Resolve each wheel by name, exactly as in the motion experiment.
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

        # Both wheel bodies in this XML have the same parent. Their positions
        # are in that shared frame, so the distance between centres is valid.
        left_body = self.model.body("wheel_left").id
        right_body = self.model.body("wheel_right").id

        if self.model.body_parentid[left_body] != self.model.body_parentid[right_body]:
            raise ValueError("Wheel separation requires a shared wheel-body parent.")

        self.wheel_separation = float(np.linalg.norm(
            self.model.body_pos[left_body] - self.model.body_pos[right_body]
        ))

        # Tyres are meshes, so their effective rolling radius is not a simple
        # XML cylinder size. Retain the 0.033 m used in our motion experiments.
        self.wheel_radius = 0.033

        # Software wheel-speed target limit from the earlier controller.
        # Torque-actuator ctrlrange is N*m and must NOT be used as a speed limit.
        self.max_wheel_speed = 25
        self.max_yaw_rate = max_yaw_rate

        # Reuse the selected PI gains. Each wheel stores its own integral.
        # Read torque limits from the compiled XML instead of duplicating them.
        self.wheel_controllers = [
            WheelPI(self.model.actuator_ctrlrange[actuator])
            for actuator in self.actuators
        ]

        self.wheel_targets = np.zeros(2)
        self.wheel_torques = np.zeros(2)

        self.viewer = None
        self.closed = False
        self.reset()

        # Passive means main retains control of stepping and update timing.
        if show_viewer:
            # Work around corrupted native UI panels on some AMD Windows drivers.
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False,
            )

            # Follow the robot with the external viewer. This does not change
            # the robot_camera used by read_camera() for localisation images.
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = self.model.body("base").id
                self.viewer.cam.distance = 2.5
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -30

    @property
    def time(self):
        """Current simulation time [s], not wall-clock time."""
        return float(self.data.time)

    @property
    def physics_dt(self):
        """Physics timestep [s], read from the loaded XML model."""
        return float(self.model.opt.timestep)

    def reset(self, pose=None):
        """Reset simulation and PI memory; optionally set (x, y, heading).

        This is experiment initialisation, not a localisation correction.
        Retain the XML's initial base height when changing planar pose.
        """
        mujoco.mj_resetData(self.model, self.data)

        if pose is not None:
            x, y, heading = pose
            start = self.base_position_index
            self.data.qpos[start:start + 2] = (x, y)

            # MuJoCo quaternions use (w, x, y, z). This is pure world-z yaw.
            self.data.qpos[start + 3:start + 7] = (
                math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)
            )

        for controller in self.wheel_controllers:
            controller.reset()

        self.wheel_targets[:] = 0.0
        self.wheel_torques[:] = 0.0

        # Recompute derived body/sensor quantities after changing joint pose.
        mujoco.mj_forward(self.model, self.data)

    def running(self):
        """Headless runs require main to provide a duration or stop condition."""
        return not self.closed and (
            self.viewer is None or self.viewer.is_running()
        )

    def read_encoders(self):
        """Return measured wheel angles [rad], speeds [rad/s] and time [s].

        Order is left then right. Both joint signs follow the current XML:
        positive wheel motion requests forward driving.
        """
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
        # Capture at the current simulation state.

        self.renderer.update_scene(self.data, camera="robot_camera")
        rgb = self.renderer.render().copy()

        return {"timestamp": self.time, "rgb": rgb}

    def read_lidar(self):
        """LiDAR sensing is not needed for the open-space controller test."""
        raise NotImplementedError("Read a timestamped LiDAR scan.")

    def apply_body_command(self, command, dt):
        """Run wheel PI once; apply torque without advancing physics.

        command is (forward speed [m/s], yaw rate [rad/s]), positive forward/left.
        dt is elapsed wheel-control time [s]. Return the limited body command.
        """
        forward_speed, yaw_rate = command

        if dt <= 0 or not np.all(np.isfinite([forward_speed, yaw_rate, dt])):
            raise ValueError("Command and dt must be finite, with dt positive.")

        yaw_rate = float(np.clip(
            yaw_rate, -self.max_yaw_rate, self.max_yaw_rate
        ))

        # A left turn needs the right wheel to move faster than the left.
        half_separation = self.wheel_separation / 2
        targets = np.array([
            (forward_speed - half_separation * yaw_rate) / self.wheel_radius,
            (forward_speed + half_separation * yaw_rate) / self.wheel_radius,
        ])

        # Scale both targets together to retain commanded curvature when a
        # wheel-speed target exceeds the software limit.
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

        # The XML has torque motors with gear=1: data.ctrl takes N*m.
        self.data.ctrl[self.actuators] = self.wheel_torques

        limited_speed = self.wheel_radius * (targets[0] + targets[1]) / 2
        limited_yaw_rate = (
            self.wheel_radius * (targets[1] - targets[0])
            / self.wheel_separation
        )
        return float(limited_speed), float(limited_yaw_rate)

    def step(self):
        """Advance one physics timestep, retaining the currently applied torque."""
        mujoco.mj_step(self.model, self.data)

    def sync_viewer(self):
        """Refresh the viewer when requested by main, independently of physics."""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_truth(self):
        """Return true (x, y, heading) for labelled diagnostics/evaluation only."""
        start = self.base_position_index
        x, y = self.data.qpos[start:start + 2]
        qw, qx, qy, qz = self.data.qpos[start + 3:start + 7]

        heading = math.atan2(
            2 * (qw * qz + qx * qy),
            1 - 2 * (qy**2 + qz**2),
        )
        return float(x), float(y), heading

    def get_truth_velocity(self):
        """Return true forward speed [m/s] and world-z yaw rate [rad/s].

        Diagnostic logging only. MuJoCo returns angular then linear velocity
        at the body frame origin, expressed here in world coordinates.
        """
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
        """Release the viewer and mark the experiment as closed."""
        self.data.ctrl[self.actuators] = 0.0

        if self.viewer is not None:
            self.viewer.close()

        self.renderer.close()
        self.closed = True

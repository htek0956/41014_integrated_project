"""Drive down the hallway, stop, turn in place, and return along its centre."""

import math

import numpy as np

from models import wrap


class HallwayReference:
    """World-frame metres/radians; +X outbound, -X on the return leg.

    The scene walls extend from x=0 to x=6. Leave 0.3 m at either end
    for the robot footprint. A turn in place needs no curved path geometry.
    """

    def __init__(self, speed):
        if speed <= 0:
            raise ValueError("Reference speed must be positive.")

        self.speed = speed
        self.start_position = np.array([0.3, 0.0])
        self.turn_position = np.array([5.7, 0.0])
        self.end_position = self.start_position.copy()
        self.start_heading = 0.0
        self.leg_length = float(self.turn_position[0] - self.start_position[0])
        self.total_length = 2 * self.leg_length

        self.arrival_tolerance = 0.03  # Along-path distance [m].
        self.deceleration = 0.3  # Reference braking envelope [m/s^2].
        self.heading_tolerance = math.radians(1.0)
        self.stopped_speed = 0.02  # Measured forward speed [m/s].
        self.stopped_yaw_rate = 0.04  # Measured turn rate [rad/s].
        self.reset()

    def reset(self):
        self.mode = "OUTBOUND"
        self.finished = False

    def _reference(self, x, heading, progress, speed, mode):
        return {
            "x": float(x),
            "y": 0.0,
            "heading": float(heading),
            "curvature": 0.0,
            "progress": float(progress),
            "speed": float(speed),
            "mode": mode,
            "stop": mode == "STOP",
        }

    def reference_at(self, progress):
        """Sample the two straight legs for plotting (the turn has no length)."""
        progress = float(np.clip(progress, 0.0, self.total_length))
        if progress < self.leg_length:
            return self._reference(
                self.start_position[0] + progress, 0.0,
                progress, self.speed, "OUTBOUND",
            )

        return self._reference(
            self.turn_position[0] - (progress - self.leg_length), math.pi,
            progress, 0.0 if progress == self.total_length else self.speed,
            "STOP" if progress == self.total_length else "RETURN",
        )

    def update(self, pose, velocity):
        """Advance from pose and measured/estimated (forward speed, yaw rate).

        Wait for braking before turning, and for heading/rate to settle before
        driving back. The current controller diagnostic supplies simulator truth;
        the assessed pipeline must supply estimates instead.
        """
        x, _, heading = pose
        speed, yaw_rate = velocity
        settled = (
            abs(speed) < self.stopped_speed
            and abs(yaw_rate) < self.stopped_yaw_rate
        )

        if self.mode == "OUTBOUND":
            if self.turn_position[0] - x <= self.arrival_tolerance:
                self.mode = "BRAKE"

        if self.mode == "BRAKE" and settled:
            self.mode = "TURN"

        if self.mode == "TURN":
            if abs(wrap(math.pi - heading)) < self.heading_tolerance and settled:
                self.mode = "RETURN"

        if self.mode == "RETURN":
            if x - self.end_position[0] <= self.arrival_tolerance:
                self.mode = "STOP"
                self.finished = True

        if self.mode in ("BRAKE", "TURN"):
            return self._reference(
                self.turn_position[0], math.pi if self.mode == "TURN" else 0.0,
                self.leg_length, 0.0, self.mode,
            )

        if self.finished:
            return self.reference_at(self.total_length)

        outbound = self.mode == "OUTBOUND"
        distance = (x - self.start_position[0]) if outbound else (self.turn_position[0] - x)
        distance = float(np.clip(distance, 0.0, self.leg_length))
        remaining = self.leg_length - distance
        reference = self.reference_at(distance if outbound else self.leg_length + distance)
        reference["speed"] = min(self.speed, math.sqrt(2 * self.deceleration * remaining))
        return reference

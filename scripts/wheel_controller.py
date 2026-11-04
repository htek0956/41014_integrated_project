"""Wheel-speed PI: speed target and measurement in, motor torque out."""


class WheelPI:
    """One controller per wheel, with its own accumulated speed correction.

    Speeds are rad/s, dt is seconds, and output torque is N*m.
    kp has units N*m/(rad/s), and ki has units N*m/rad.
    Set ki=0 for proportional control.
    """

    def __init__(self, torque_limits, kp=.25, ki=0.25):
        self.kp = kp
        self.ki = ki
        self.min_torque, self.max_torque = torque_limits

        if min(kp, ki) < 0:
            raise ValueError("Gains must be nonnegative.")

        if self.min_torque >= self.max_torque:
            raise ValueError("Torque limits must be ordered: minimum, maximum.")

        self.reset()

    def reset(self):
        """Clear memory when starting a new experiment or resetting the robot."""
        # Store the integral contribution as torque so it is easy to inspect.
        self.integral_torque = 0.0
        self.raw_torque = 0.0

    def update(self, target_speed, measured_speed, dt):
        """Advance once per control interval and return limited wheel torque."""
        if dt <= 0:
            raise ValueError("dt must be positive.")

        error = target_speed - measured_speed

        # P responds straight away: a bigger speed error asks for more torque.
        proportional_torque = self.kp * error

        # I gradually builds the extra torque needed to overcome persistent
        # resistance. First try the update, then check whether it would wind up.
        integral_change = self.ki * error * dt
        candidate_integral = self.integral_torque + integral_change
        candidate_torque = proportional_torque + candidate_integral

        # Don't keep asking for more when the motor is already at its limit.
        # Still allow the stored correction to unwind in the other direction.
        pushes_above_limit = (
            candidate_torque > self.max_torque and integral_change > 0
        )
        pushes_below_limit = (
            candidate_torque < self.min_torque and integral_change < 0
        )

        if not (pushes_above_limit or pushes_below_limit):
            self.integral_torque = candidate_integral

        self.raw_torque = proportional_torque + self.integral_torque

        return max(self.min_torque, min(self.raw_torque, self.max_torque))

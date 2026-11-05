class WheelPI:


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


        self.integral_torque = 0.0
        self.raw_torque = 0.0

    def update(self, target_speed, measured_speed, dt):

        if dt <= 0:
            raise ValueError("dt must be positive.")

        error = target_speed - measured_speed


        proportional_torque = self.kp * error


        integral_change = self.ki * error * dt
        candidate_integral = self.integral_torque + integral_change
        candidate_torque = proportional_torque + candidate_integral


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

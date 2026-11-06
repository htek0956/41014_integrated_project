import numpy as np
import models

class PoseEKF:


    def __init__(self, initial_pose, initial_covariance, geometry):

        self.state = np.array(initial_pose, dtype=float)
        self.covariance = np.array(initial_covariance, dtype=float)
        self.geometry = geometry

        # One-step prediction residual covariance from process_noise/run_002.
        # Per 0.01 s prediction, in world x/y (m^2) and heading (rad^2).
        # The factor 8 is empirical dropout tuning, not sensor calibration.

        self.Q = np.diag([
            2.42388189e-10,
            2.80857506e-10,
            1.16930470e-8,
        ])

        self.Q = 8 * self.Q

        self.previous_left_angle = None
        self.previous_right_angle = None


    def predict(self, encoder_measurement):

        new_left = encoder_measurement["left_angle"]
        new_right = encoder_measurement["right_angle"]

        if self.previous_left_angle is None:
            self.previous_left_angle = new_left
            self.previous_right_angle = new_right
            return

        delta_left = new_left - self.previous_left_angle
        delta_right = new_right - self.previous_right_angle

        delta_s, delta_theta = models.wheel_increments_to_motion(delta_left, delta_right, self.geometry)

        x_pred = models.propagate_pose(self.state, (delta_s, delta_theta))
        F = models.motion_jacobians(self.state, (delta_s, delta_theta))

        P_pred = F @ self.covariance @ F.T + self.Q

        self.state = np.array(x_pred, dtype=float)
        self.covariance = P_pred

        self.previous_left_angle = new_left
        self.previous_right_angle = new_right

    def correct(self, camera_observation):

        if not camera_observation["valid"]:
            return
        z = np.asarray(camera_observation["pose"], dtype=float)
        R = np.asarray(camera_observation["covariance"], dtype=float)

        innovation = z - self.state
        innovation[2] = models.wrap(innovation[2])

        S = self.covariance + R
        #Solve for kalman gain to best update state estimate
        #Using S as the covariance of innovation and the covariance of the state to find the optimal gain
        K = np.linalg.solve(S.T, self.covariance.T).T

        state_correction = K @ innovation

        self.state += state_correction
        # Wrap the corrected heading, not the adjustment applied to it.
        self.state[2] = models.wrap(self.state[2])

        I = np.eye(3)
        self.covariance = (I - K) @ self.covariance
        self.covariance = (self.covariance + self.covariance.T) / 2

    def get_estimate(self):

        return {
            "pose": tuple(self.state),
            "covariance": self.covariance.copy(),
        }

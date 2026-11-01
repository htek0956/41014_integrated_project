"""Estimated-state path controller and tracking-error geometry."""

import control as ct
import numpy as np

from models import wrap


class Controller:
    """Outer estimated-state tracker. Wheel PI belongs to the robot interface."""

    def __init__(self, speed, dt):
               
        self.K_lqr = calculate_lqr_gain(speed, 0.0, dt)  # Straight path
    

    def update(self, estimate, reference,  dt):
        """Return (v, omega) in m/s and rad/s, positive forward/left.

        Use estimated pose, curvature feedforward and error feedback.
        Handle alignment and stopping separately from moving-path tracking.
        dt is the elapsed outer-control interval in seconds.
        """
        e_y, e_theta = tracking_errors(estimate, reference)

        e = np.array([[e_y],
                      [e_theta]])

        v = reference["speed"]
        kappa = reference["curvature"]

        feedback = (self.K_lqr @ e).item()
        command_omega = v * kappa - feedback

        return v, command_omega

def tracking_errors(pose, reference):
    """Return signed cross-track (m) and wrapped heading error (rad).

    pose is (x, y, heading) in world-frame metres/metres/radians.
    reference is the dictionary returned by PaperclipReference.update().
    Positive cross-track means left of the path direction. Positive heading
    error means the robot points counterclockwise from the path tangent.
    """
    x, y, heading = pose

    offset_x = x - reference["x"]
    offset_y = y - reference["y"]
    path_heading = reference["heading"]

    # Project the position offset onto the path's left-pointing unit normal.
    # This works for both straights and arcs when the reference point is the
    # corresponding projection onto the path, as in Assignment 3.
    cross_track_error = (
        -np.sin(path_heading) * offset_x
        + np.cos(path_heading) * offset_y
    )

    heading_error = wrap(heading - path_heading)

    return float(cross_track_error), float(heading_error)


def calculate_lqr_gain(v, kappa, Ts, q_e_y=0.02, q_e_theta=0.1, r_d_omega=0.26):

    # Discrete linearised path-following error model.
    # State e = [e_y, e_theta], input d_omega = omega - v*kappa. Returns (Ad, Bd).

    # 1. Continuous Jacobians A (2x2) and B (2x1) at e = 0, d_omega = 0

    # Done with hand caluclations.
    # Derived by hand by differentiating the error dynamics and evaluating at
    # e_y = 0, e_theta = 0, and delta_omega = 0.

    A = np.array([
        [0, v],
        [-v * kappa**2, 0],
    ], dtype=float)

    #Steering input only rotational error so taking the jacobian,
    B = np.array([
        [0],
        [1],
    ], dtype=float)
    
    # 2. Zero-order-hold discretisation at Ts
    discrete = ct.c2d(ct.ss(A, B, np.zeros((1,2)), np.zeros((1,1))), Ts, method="zoh")
    A = discrete.A
    B = discrete.B

    # Choose Q_lqr (2x2) and R_lqr (1x1), compute K_lqr (1x2) with the discrete LQR,

    # Using Bryson's Rule, the weights are based on chosen error/control scales.
    Q_lqr = np.diag([1 / q_e_y**2, 1 / q_e_theta**2])
    R_lqr = np.array([[1 / r_d_omega**2]])

    K_lqr, _, _ = ct.dlqr(A, B, Q_lqr, R_lqr)
    K_lqr = np.array(K_lqr)

    # Closed-loop eigenvalues: eig(Ad - Bd * K)
    poles = np.linalg.eigvals(A - B @ K_lqr)

    print(f"K_lqr: {K_lqr}")
    print(f"Design-model CL eigenvalue magnitudes: {np.abs(poles)}")

    return K_lqr

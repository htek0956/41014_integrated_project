import control as ct
import numpy as np

from models import wrap


class Controller:


    def __init__(self, speed, dt, verbose=True):

        self.K_lqr = calculate_lqr_gain(speed, 0.0, dt, verbose=verbose)


    def update(self, estimate, reference,  dt):


        mode = reference.get("mode")
        if reference["stop"] or mode == "BRAKE":
            return 0.0, 0.0

        if mode == "TURN":
            heading_error = wrap(reference["heading"] - estimate[2])
            return 0.0, float(np.clip(2.0 * heading_error, -0.8, 0.8))

        e_y, e_theta = tracking_errors(estimate, reference)

        e = np.array([[e_y],
                      [e_theta]])

        v = reference["speed"]
        kappa = reference["curvature"]

        feedback = (self.K_lqr @ e).item()
        command_omega = v * kappa - feedback

        return v, command_omega

def tracking_errors(pose, reference):

    x, y, heading = pose

    offset_x = x - reference["x"]
    offset_y = y - reference["y"]
    path_heading = reference["heading"]


    cross_track_error = (
        -np.sin(path_heading) * offset_x
        + np.cos(path_heading) * offset_y
    )

    heading_error = wrap(heading - path_heading)

    return float(cross_track_error), float(heading_error)


def calculate_lqr_gain(v, kappa, Ts, q_e_y=0.02, q_e_theta=0.1, r_d_omega=0.26, verbose=True):


    A = np.array([
        [0, v],
        [-v * kappa**2, 0],
    ], dtype=float)


    B = np.array([
        [0],
        [1],
    ], dtype=float)


    discrete = ct.c2d(ct.ss(A, B, np.zeros((1,2)), np.zeros((1,1))), Ts, method="zoh")
    A = discrete.A
    B = discrete.B


    Q_lqr = np.diag([1 / q_e_y**2, 1 / q_e_theta**2])
    R_lqr = np.array([[1 / r_d_omega**2]])

    K_lqr, _, _ = ct.dlqr(A, B, Q_lqr, R_lqr)
    K_lqr = np.array(K_lqr)


    poles = np.linalg.eigvals(A - B @ K_lqr)

    if verbose:
        print(f"K_lqr: {K_lqr}")
        print(f"Design-model CL eigenvalue magnitudes: {np.abs(poles)}")

    return K_lqr

"""Simple Part A rank checks, following Assessment 3 Task A.1.

Run from the project folder: python scripts/model_analysis.py
"""

import control as ct
import numpy as np

from models import motion_jacobians, wheel_increments_to_motion


def main():

    np.set_printoptions(precision=6, suppress=True)

    # Match REFERENCE_SPEED and CONTROL_DT in main.py without importing MuJoCo.
    speed = 0.6                  # m/s
    Ts = 0.01                    # s
    geometry = {"wheel_radius": 0.033, "wheel_base": 0.288}

    print("Geometry:", geometry, "(metres)")

    # Same tracking model and ZOH discretisation as calculate_lqr_gain().
    # States: [cross-track error (m), heading error (rad)].
    # Input: delta_omega = omega - v*kappa (rad/s).
    for name, v, kappa in [("Straight", speed, 0.0),
                           ("Turn (example radius 1 m)", speed, 1.0),
                           ("Stopped", 0.0, 0.0)]:

        A = np.array([[0.0, v], [-v * kappa**2, 0.0]])
        B = np.array([[0.0], [1.0]])

        discrete = ct.c2d(
            ct.ss(A, B, np.zeros((1, 2)), np.zeros((1, 1))),
            Ts, method="zoh")

        C_cont = ct.ctrb(A, B)
        C_disc = ct.ctrb(discrete.A, discrete.B)
        rank_cont = np.linalg.matrix_rank(C_cont)
        rank_disc = np.linalg.matrix_rank(C_disc)

        print(f"\n{name}: v={v} m/s, kappa={kappa} 1/m")
        print("Continuous controllability matrix:\n", C_cont)
        print("Discrete controllability matrix:\n", C_disc)
        print(f"Controllability ranks: continuous {rank_cont}/2, discrete {rank_disc}/2")

        expected = 2 if v != 0.0 else 1
        assert rank_cont == expected and rank_disc == expected

    # Use the implemented EKF transition Jacobian at an example straight step.
    # The encoder increments are known inputs, not absolute pose observations.
    wheel_increment = speed * Ts / geometry["wheel_radius"]
    motion = wheel_increments_to_motion(wheel_increment, wheel_increment, geometry)
    F = motion_jacobians([0.0, 0.0, 0.0], motion)
    H = np.eye(3)                # Valid world-frame camera pose: z = pose + noise.
    O = ct.obsv(F, H)
    rank_camera = np.linalg.matrix_rank(O)

    print("\nPose observability with a valid camera measurement")
    print("F:\n", F)
    print("H:\n", H)
    print("Observability matrix [H; HF; HF^2]:\n", O)
    print(f"Observability rank: {rank_camera}/3")
    assert rank_camera == 3

    # Zero H represents no pose information throughout a camera outage.
    # The real EKF skips correction; it does not inject a zero measurement.
    O_loss = ct.obsv(F, np.zeros((3, 3)))
    rank_loss = np.linalg.matrix_rank(O_loss)
    print(f"\nWithout camera pose updates: observability rank {rank_loss}/3")
    print("Encoders predict relative motion but cannot correct absolute pose drift.")
    assert rank_loss == 0

    print("\nAll expected rank checks passed.")


if __name__ == "__main__":
    main()

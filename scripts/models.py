import numpy as np


def wrap(angle):

    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def encoder_angle_measurement(angle, counts_per_revolution=4096):

    # The Waffle Pi wheel motor has a 12-bit position encoder: 4096 counts
    # per revolution. This models its resolution not extra random jitter.
    # Source: https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_dynamixel/
    if (not np.isfinite(counts_per_revolution)
            or counts_per_revolution <= 0
            or counts_per_revolution != int(counts_per_revolution)):
        raise ValueError("Encoder counts per revolution must be a positive integer.")

    angle = np.asarray(angle, dtype=float)
    if not np.all(np.isfinite(angle)):
        raise ValueError("Encoder angles must be finite, in radians.")

    angle_per_count = 2 * np.pi / counts_per_revolution

    # Round the absolute wheel angle to the nearest count. Keep multiple
    # revolutions and reverse travel; wrapping would create false increments.
    measured_angle = np.round(angle / angle_per_count) * angle_per_count

    return measured_angle


# Convert wheel-angle increments [rad] to axle travel [m] and heading change [rad].
def wheel_increments_to_motion(delta_left, delta_right, geometry):

    wheel_base = geometry["wheel_base"]
    wheel_radius = geometry["wheel_radius"]

    s_left = delta_left * wheel_radius
    s_right = delta_right * wheel_radius

    delta_s = (s_left + s_right) / 2.0
    delta_theta = (s_right - s_left) / wheel_base

    return delta_s, delta_theta


def propagate_pose(pose, motion):

    x, y, heading = pose
    delta_s, delta_theta = motion

    new_heading = wrap(heading + delta_theta)

    # For a very small turn, use the straight-line limit rather than divide
    # by a near-zero heading change. Motion already includes the timestep.
    if abs(delta_theta) < 1e-6:
        new_x = x + delta_s * np.cos(heading)
        new_y = y + delta_s * np.sin(heading)

    # Otherwise assume constant curvature over this measurement interval.
    else:
        radius = delta_s / delta_theta

        new_x = x + radius * (np.sin(new_heading) - np.sin(heading))
        new_y = y + radius * (np.cos(heading) - np.cos(new_heading))

    return float(new_x), float(new_y), float(new_heading)


def motion_jacobians(pose, motion):

    x, y, heading = pose
    delta_s, delta_theta = motion

    F = np.eye(3)

    # State Jacobian df/d[x, y, heading], evaluated at the starting pose.
    # Treat the measured distance and heading change as constants here.
    if abs(delta_theta) < 1e-6:

        F[0, 2] = -delta_s * np.sin(heading)
        F[1, 2] = delta_s * np.cos(heading)

    else:
        radius = delta_s / delta_theta

        F[0, 2] = radius * (np.cos(heading + delta_theta) - np.cos(heading))
        F[1, 2] = radius * (np.sin(heading + delta_theta) - np.sin(heading))

    return F


if __name__ == "__main__":

    checks = [
        ("No motion", [0, 0, 0], [0, 0], np.eye(3)),
        ("Forward 1 m", [0, 0, 0], [1, 0],
         np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])),
        ("Forward 1 m facing +Y", [0, 0, np.pi / 2], [1, 0],
         np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]])),
        ("Left quarter-circle, radius 1 m", [0, 0, 0], [np.pi / 2, np.pi / 2],
         np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]])),
    ]

    for name, pose, motion, expected in checks:

        actual = motion_jacobians(pose, motion)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, err_msg=name)
        print(f"PASS: {name}")

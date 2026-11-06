"""Predict collisions using LiDAR points relative to the current robot position"""

# Load numpy for working with arrays and distances
import numpy as np

# Reuse the existing model for predicting robot movement
from models import propagate_pose


# Accept the scan and proposed command with adjustable prediction settings
# Use a circular robot footprint and include the LiDAR mounting offset
def check_collision(scan, command, horizon=1.0, prediction_dt=0.05,
                    robot_radius=0.15, margin=0.05, lidar_offset=(-0.049, 0.0),
                    scan_pose=None, current_pose=None):
    """Predict motion at constant forward speed and turn rate

    Distances are metres and times are seconds and angles are radians
    The circle is centred on the wheel axle
    The planner currently uses a larger radius and margin than this motion check
    The LiDAR faces forward with positive x forward and positive y left

    Only detected points are checked and invalid readings remain unknown space
    Braking and moving obstacles are not modelled and the EKF is not changed
    """

    # Identify ranges that contain a finite measured distance
    valid = np.isfinite(scan["ranges"])
    # Keep only those measured distances
    ranges = scan["ranges"][valid]
    # Keep the beam angle belonging to each retained distance
    angles = scan["angles"][valid]
    # Convert each distance and angle into forward and left coordinates
    obstacles = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
    # Shift the points from the sensor origin to the robot axle origin
    obstacles += np.asarray(lidar_offset)

    # Rejoining can happen between scans so express saved returns relative to the current axle
    # Both poses are EKF estimates and no simulator truth is used for this frame change
    if scan_pose is not None or current_pose is not None:
        if scan_pose is None or current_pose is None:
            raise ValueError('Collision frame conversion needs both scan and current poses')

        scan_x, scan_y, scan_heading = scan_pose
        x, y, heading = current_pose
        c, s = np.cos(scan_heading), np.sin(scan_heading)
        scan_rotation = np.array([[c, -s], [s, c]])
        world_points = obstacles @ scan_rotation.T + np.array([scan_x, scan_y])

        # Row vectors use the rotation without transpose when going from world back to robot
        c, s = np.cos(heading), np.sin(heading)
        current_rotation = np.array([[c, -s], [s, c]])
        obstacles = (world_points - np.array([x, y])) @ current_rotation

    # Read the proposed forward speed and turn rate
    v, omega = command
    # Start the prediction at the current robot origin with zero heading
    pose = (0.0, 0.0, 0.0)
    # Create a list to store the predicted positions and headings
    predicted_path = []
    # Start with no predicted collision time
    collision_time = None
    # Keep the first predicted pose where an obstacle enters the footprint
    collision_pose = None

    # Round up the number of prediction steps to cover the full horizon
    steps = int(np.ceil(horizon / prediction_dt))
    # Adjust the step duration so the prediction ends exactly at the horizon
    dt = horizon / steps
    # Check the current pose and every future pose including the final pose
    for step in range(steps + 1):
        # Save this predicted pose for plotting or inspection
        predicted_path.append(pose)
        # Calculate the distance from this predicted robot centre to every obstacle point
        distances = np.linalg.norm(obstacles - np.asarray(pose[:2]), axis=1)
        # Record only the first pose where a point enters the circle including its margin
        if collision_time is None and np.any(distances <= robot_radius + margin):
            # Convert the prediction step number into seconds from now
            collision_time = step * dt
            # Save the pose so the plot can place the collision circle correctly
            collision_pose = pose

        # Move to the next pose unless this is already the final pose
        if step < steps:
            # Convert speed into travel distance and turn rate into heading change
            pose = propagate_pose(pose, (v * dt, omega * dt))

    # Return the collision decision and the prediction details
    return {
        # Report whether any checked pose overlaps a detected obstacle point
        "collision": collision_time is not None,
        # Return the first collision time or None when no collision was predicted
        "collision_time": collision_time,
        # Return the first collision pose or None when no collision was predicted
        "collision_pose": collision_pose,
        # Convert the stored poses into a numpy array
        "predicted_path": np.asarray(predicted_path),
        # Share the checked circle size and mounting offset with the plot
        "footprint_radius": robot_radius + margin,
        "lidar_offset": tuple(lidar_offset),
    }

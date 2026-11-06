"""Choose whether to track the route or hold the robot stopped

Main uses this state machine during route tracking
"""

import numpy as np

from path_planning import astar, OpenPathReference


class Navigation:

    def __init__(self):

        # Remember the operating state between calls
        self.state = "TRACK"

        # Count consecutive clear scans rather than repeated controller updates
        self.clear_scans = 0
        self.required_clear_scans = 3

        # Keep only one fresh scan per confirmed stop to limit permanent noise thickening
        self.obstacle_mapped = False

        # Keep the original path separate from the temporary avoidance reference
        self.detour = None
        self.rejoin_progress = None
        self.planning_attempted = False
        self.rejoined = False

        # Provisional thresholds for deciding when physical motion has stopped
        # Forward speed is metres per second and yaw rate is radians per second
        self.stopped_speed = 0.015
        self.stopped_yaw_rate = 0.04


    def update(self, proposed_command, collision, forward_speed, yaw_rate, new_scan):
        """Choose the executed command using the current navigation state

        proposed_command is the tracker command even when the robot is stopped
        collision is the result of checking that proposed command against a scan
        forward_speed and yaw_rate are measured from the wheel encoders
        new_scan is true only when a fresh LiDAR scan and collision result arrive

        Return forward speed and yaw rate to execute
        No simulator truth is needed here
        """

        if self.state in ("TRACK", "AVOID"):

            # Continue tracking while the proposed motion has no detected collision
            if not collision["collision"]:
                return proposed_command
            else:
                # Request braking immediately and forget any previous clear scans
                self.state = "STOPPING"
                self.clear_scans = 0
                self.obstacle_mapped = False

                # A new blockage needs a new stopped scan and a new planning attempt
                # Original route progress stays frozen if the blockage occurs on a detour
                self.detour = None
                self.rejoin_progress = None
                self.planning_attempted = False

        elif self.state == "STOPPING":

            # Confirm the wheels have settled rather than assuming zero command means zero motion
            if abs(forward_speed) < self.stopped_speed and abs(yaw_rate) < self.stopped_yaw_rate:
                self.state = "BLOCKED"

        elif self.state == "BLOCKED":

            # A failed plan must remain stopped even if the old proposal later appears clear
            if self.detour is None:
                return self.stop()

            # Count each scan once even when this method runs faster than the LiDAR
            if new_scan:
                if collision["collision"]:
                    self.clear_scans = 0
                else:
                    self.clear_scans += 1

            # Resume only after several fresh scans agree that the proposed motion is clear
            if self.clear_scans >= self.required_clear_scans:
                self.state = "AVOID"
                self.clear_scans = 0
                return proposed_command

        # Keep requesting zero wheel targets throughout stopping and waiting
        return self.stop()


    def plan_detour(self, path, pose, rejoin_distance):
        """Try one A star detour after the stopped scan has updated the map"""

        if self.state != "BLOCKED" or not self.obstacle_mapped or self.planning_attempted:
            return

        # Try once per stop rather than repeatedly planning from the same noisy scan
        self.planning_attempted = True
        self.clear_scans = 0
        rejoin_progress = path.progress + rejoin_distance

        try:
            # Do not wrap around the loop or move the single target if it is unavailable
            # A moving rejoin needs a point before the original route stopping endpoint
            if rejoin_progress >= path.total_length:
                raise ValueError('Rejoin target reaches beyond the moving part of the original route')

            rejoin = path.reference_at(rejoin_progress)
            goal = np.array([rejoin["x"], rejoin["y"]])
            grid = path.grid
            route = astar(grid, grid.cells(pose[:2]), grid.cells(goal))

            # A star returns cell centres so include the exact robot and original route positions
            # The open reference checks the resulting smoothed curve against inflated clearance
            route = np.vstack((pose[:2], route, goal))
            detour = OpenPathReference(
                grid, route, rejoin["heading"], path.speed, rejoin["speed"])

        except ValueError as error:
            # Leave detour unset so the command gate keeps requesting zero wheel speeds
            print(f'Detour unavailable and robot remains stopped: {error}')
            return

        self.detour = detour
        self.rejoin_progress = rejoin_progress
        print(f'Detour planned to original route progress {rejoin_progress:.2f} m')


    def update_reference(self, path, pose):
        """Select the reference while keeping original and detour progress separate"""

        # Main uses this flag to collision check the original route command at handover
        self.rejoined = False

        if self.detour is not None:

            if self.state == "AVOID":
                reference = self.detour.update(pose)

                if self.detour.finished:
                    # Explicitly restore the saved original progress instead of searching from the old stop
                    path.progress = self.rejoin_progress
                    self.detour = None
                    self.state = "TRACK"
                    self.rejoined = True
                    return path.reference_at(path.progress)

                return reference

            # Hold detour progress while three fresh scans check the proposed restart command
            return self.detour.reference_at(self.detour.progress)

        if self.state == "TRACK":
            return path.update(pose)

        # Braking and blocked states must not advance the original route
        return path.reference_at(path.progress)


    def update_obstacle_map(self, grid, scan, scan_pose, lidar_offset):
        """Add one fresh scan after stopping using its matching EKF pose

        Main calls this only when a new scan arrives
        Scan pose is world x and y in metres and heading in radians
        LiDAR offset is measured from the axle in the robot forward and left frame
        """

        # BLOCKED means the measured wheel motion has already confirmed the stop
        # Do not accumulate more scans while waiting at the same blockage
        if self.state != "BLOCKED" or self.obstacle_mapped:
            return

        # Keep every valid measured endpoint including wall returns
        # Missing returns provide no occupied cell and do not clear any old detections
        valid = np.isfinite(scan["ranges"]) & np.isfinite(scan["angles"])
        ranges = scan["ranges"][valid]
        angles = scan["angles"][valid]

        # First move from the LiDAR origin to the robot axle origin
        points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        points += np.asarray(lidar_offset)

        # Rotate forward and left coordinates into world coordinates then add axle position
        # Use the saved scan pose rather than the later pose when mapping is requested
        x, y, heading = scan_pose
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        world_points = points @ rotation.T + np.array([x, y])

        # Occupied cells persist and the grid refreshes its clearance and inflated safe cells
        grid.add_obstacles(world_points)
        self.obstacle_mapped = True


    def stop(self):
        """Return the body command used to request a stop

        Wheel PI should remain active to bring measured wheel speeds to zero
        Requesting a stop is different from confirming the robot has stopped
        """

        # Request zero forward speed and zero turn rate while keeping wheel PI active
        return 0.0, 0.0

"""Live scan and predicted motion relative to the current robot axle"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


class LidarPlot:
    def __init__(self):
        self.figure, self.axis = plt.subplots(figsize=(6, 6))
        self.figure.canvas.manager.set_window_title('Robot LiDAR')
        self.points = self.axis.scatter([], [], s=8, label='Range returns')
        self.axis.plot(0, 0, 'k>', label='Robot axle', markersize=8)
        self.sensor, = self.axis.plot([], [], 'rx', label='LiDAR origin')
        self.path, = self.axis.plot([], [], 'g-', label='Predicted path', linewidth=2)
        # Show the checked footprint at the first collision or the final predicted pose
        self.footprint = Circle((0, 0), 0, fill=False, linestyle='--',
                                label='Predicted footprint plus margin')
        self.axis.add_patch(self.footprint)
        self.axis.set(xlabel='Forward from axle [m]', ylabel='Left from axle [m]', aspect='equal')
        self.axis.grid(True, alpha=.3)
        self.axis.legend(loc='upper right')
        plt.show(block=False)

    def update(self, scan, collision):
        if not plt.fignum_exists(self.figure.number):
            return
        valid = np.isfinite(scan['ranges'])
        ranges, angles = scan['ranges'][valid], scan['angles'][valid]
        # Put obstacle points and the predicted path in the same axle centred frame
        offset = np.asarray(collision['lidar_offset'])
        points = np.column_stack((ranges*np.cos(angles), ranges*np.sin(angles))) + offset
        self.points.set_offsets(points)
        self.sensor.set_data([offset[0]], [offset[1]])

        path = collision['predicted_path']
        color = 'red' if collision['collision'] else 'green'
        self.path.set_data(path[:, 0], path[:, 1])
        self.path.set_color(color)
        pose = collision['collision_pose'] if collision['collision'] else path[-1]
        self.footprint.center = pose[:2]
        self.footprint.set_radius(collision['footprint_radius'])
        self.footprint.set_edgecolor(color)
        status = (f"Collision in {collision['collision_time']:.2f} s"
                  if collision['collision'] else 'No detected collision')
        limit = scan['range_max']
        self.axis.set(xlim=(-limit, limit), ylim=(-limit, limit),
                      title=f"2D LiDAR | t = {scan['timestamp']:.1f} s\n{status}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)

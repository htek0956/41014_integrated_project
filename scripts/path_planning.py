import argparse
import heapq
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label
from scipy.interpolate import CubicSpline

DEFAULT_SCENE = Path(__file__).resolve().parents[1] / 'robotis_tb3' / 'scene_loop.xml'


class OccupancyMap:


    def __init__(self, occupied, origin, resolution, robot_radius=.25, margin=.12):
        self.occupied = np.asarray(occupied, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = float(resolution)
        self.robot_radius = robot_radius
        self.margin = margin

        self.clearance = np.maximum(0, distance_transform_edt(~self.occupied)
                                    * resolution - resolution / np.sqrt(2))
        self.safe = self.clearance > robot_radius + margin

    def cells(self, points):
        return np.rint((np.asarray(points) - self.origin) / self.resolution).astype(int)

    def world(self, cells):
        return self.origin + np.asarray(cells) * self.resolution

    def clearances(self, points):
        cells = self.cells(points)
        x, y = cells[..., 0], cells[..., 1]
        valid = (x >= 0) & (y >= 0) & (x < self.safe.shape[1]) & (y < self.safe.shape[0])
        result = np.zeros(x.shape)
        result[valid] = self.clearance[y[valid], x[valid]]
        return result


def map_from_scene(scene=DEFAULT_SCENE, resolution=.05):

    root = ET.parse(scene).getroot()
    walls = []
    for geom in root.findall('./worldbody/geom'):
        if not geom.get('name', '').startswith('nav_wall_'):
            continue
        if geom.get('type') != 'box' or any(k in geom.attrib for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Ground-truth mapper requires unrotated wall boxes.')
        centre = np.fromstring(geom.get('pos'), sep=' ')[:2]
        size = np.fromstring(geom.get('size'), sep=' ')[:2]
        walls.append((centre, size))
    if not walls:
        raise ValueError('No nav_wall_ geometry in scene.')
    start_site = root.find("./worldbody/site[@name='navigation_start']")
    if start_site is None:
        raise ValueError('Scene needs a navigation_start site.')
    start = np.fromstring(start_site.get('pos'), sep=' ')[:2]
    lower = np.floor(np.min([p-s for p, s in walls], axis=0) - 1)
    upper = np.ceil(np.max([p+s for p, s in walls], axis=0) + 1)
    x = np.arange(lower[0], upper[0] + resolution/2, resolution)
    y = np.arange(lower[1], upper[1] + resolution/2, resolution)
    xx, yy = np.meshgrid(x, y)
    occupied = np.zeros(xx.shape, dtype=bool)
    for p, s in walls:
        occupied |= (abs(xx-p[0]) <= s[0]+resolution/2) & (abs(yy-p[1]) <= s[1]+resolution/2)


    components, _ = label(~occupied)
    cell = np.rint((start-lower)/resolution).astype(int)
    region = components[cell[1], cell[0]]
    if region == 0:
        raise ValueError('Start is occupied.')
    free = components == region
    if free[0].any() or free[-1].any() or free[:, 0].any() or free[:, -1].any():
        raise ValueError('Corridor is not enclosed: start connects to map exterior.')
    return OccupancyMap(~free, lower, resolution), start


def astar(grid, start, goal, traversable=None):

    safe = grid.safe if traversable is None else traversable
    start, goal = tuple(start), tuple(goal)
    for x, y in (start, goal):
        if not (0 <= y < safe.shape[0] and 0 <= x < safe.shape[1] and safe[y, x]):
            raise ValueError('A* endpoint is outside safe free space.')
    queue = [(0., start)]
    costs, parent = {start: 0.}, {}
    closed = set()
    while queue:
        _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            route = [current]
            while route[-1] != start:
                route.append(parent[route[-1]])
            return grid.world(route[::-1])
        closed.add(current)
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x+dx, y+dy
            if not (0 <= ny < safe.shape[0] and 0 <= nx < safe.shape[1] and safe[ny, nx]):
                continue
            if dx and dy and not (safe[y, nx] and safe[ny, x]):
                continue

            step = np.hypot(dx, dy) * grid.resolution * (1 + .8 / grid.clearance[ny, nx]**2)
            candidate = costs[current] + step
            node = (nx, ny)
            if candidate < costs.get(node, float('inf')):
                costs[node], parent[node] = candidate, current
                heuristic = np.hypot(nx-goal[0], ny-goal[1]) * grid.resolution
                heapq.heappush(queue, (candidate + heuristic, node))
    raise ValueError('No safe route. Corridor may be blocked or too narrow.')


class PlannedLoopReference:


    def __init__(self, grid, start, speed=.6):
        if speed <= 0:
            raise ValueError('Speed must be positive.')
        self.grid, self.speed = grid, speed
        self.start_position = np.asarray(start, dtype=float)
        self.end_position = self.start_position.copy()

        ys, xs = np.nonzero(grid.safe)
        points = grid.world(np.column_stack((xs, ys)))
        nearby = points[np.linalg.norm(points-start, axis=1) < 1.2]
        _, axes = np.linalg.eigh(np.cov(nearby.T))
        tangent = axes[:, -1]
        if tangent[np.argmax(abs(tangent))] < 0:
            tangent = -tangent
        offsets = points-start
        along = offsets @ tangent
        across = offsets @ np.array([-tangent[1], tangent[0]])
        cut = (abs(along) < .10) & (abs(across) < 1.5)
        safe = grid.safe.copy()
        safe[ys[cut], xs[cut]] = False
        entry, exit = grid.cells(start + .20*tangent), grid.cells(start - .20*tangent)
        route = astar(grid, entry, exit, safe)
        route = np.vstack((start, route, start))
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]


        spacing = .025
        samples = np.arange(0, distances[-1], spacing)
        dense = np.column_stack([np.interp(samples, distances, route[:, j]) for j in (0, 1)])
        for smoothing in (.45, .35, .25, .15):
            smooth = gaussian_filter1d(dense, smoothing/spacing, axis=0, mode='wrap')

            smooth[0] = start
            smooth = np.vstack((smooth, smooth[0]))
            s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
            spline = CubicSpline(s, smooth, bc_type='periodic')
            check_s = np.linspace(0, s[-1], int(s[-1]/.01)+1)
            if np.min(grid.clearances(spline(check_s))) > grid.robot_radius + grid.margin:
                break
        else:
            raise ValueError('Smoothing cannot preserve footprint clearance.')
        self.spline, self.total_length = spline, float(s[-1])
        self.sample_s = np.linspace(0, self.total_length, int(self.total_length/.025)+1)
        self.sample_xy = spline(self.sample_s)
        self.progress = 0.
        self.finished = False
        self.start_heading = self.reference_at(0)['heading']

    def reference_at(self, progress):
        progress = float(np.clip(progress, 0, self.total_length))
        point = self.spline(progress)
        d, dd = self.spline(progress, 1), self.spline(progress, 2)
        curvature = float((d[0]*dd[1]-d[1]*dd[0]) / np.linalg.norm(d)**3)

        speed = min(self.speed, .65/max(abs(curvature), 1e-6),
                    np.sqrt(.25/max(abs(curvature), 1e-6)))

        ahead = np.minimum(progress + np.linspace(0, 1.5, 31), self.total_length)
        ad, add = self.spline(ahead, 1), self.spline(ahead, 2)
        k = abs(ad[:, 0]*add[:, 1]-ad[:, 1]*add[:, 0]) / np.linalg.norm(ad, axis=1)**3
        limits = np.minimum(self.speed, np.minimum(.65/np.maximum(k, 1e-6), np.sqrt(.25/np.maximum(k, 1e-6))))
        speed = min(speed, float(np.min(np.sqrt(limits**2 + 2*.25*(ahead-progress)))),
                    np.sqrt(2*.25*max(0., self.total_length-progress)))
        return dict(x=float(point[0]), y=float(point[1]), heading=float(np.arctan2(d[1], d[0])),
                    curvature=curvature, progress=progress, speed=float(speed),
                    mode='STOP' if self.finished else 'NAVIGATE', stop=self.finished)

    def update(self, pose, velocity=None):


        lower = max(0, np.searchsorted(self.sample_s, self.progress-.10))
        upper = min(len(self.sample_s), np.searchsorted(self.sample_s, self.progress+.75)+1)
        index = lower + np.argmin(np.linalg.norm(self.sample_xy[lower:upper]-np.asarray(pose[:2]), axis=1))
        self.progress = max(self.progress, float(self.sample_s[index]))
        if self.total_length-self.progress < .04 and np.linalg.norm(np.asarray(pose[:2])-self.end_position) < .06:
            self.finished = True
        return self.reference_at(self.progress)


def plot_plan(path, output=None):
    import matplotlib.pyplot as plt
    grid = path.grid
    fig, ax = plt.subplots(figsize=(11, 9))
    end = grid.world((grid.safe.shape[1]-1, grid.safe.shape[0]-1))
    ax.imshow(grid.occupied, origin='lower', extent=[grid.origin[0], end[0], grid.origin[1], end[1]], cmap='Greys', vmin=0, vmax=1)
    ax.plot(path.sample_xy[:, 0], path.sample_xy[:, 1], color='tab:blue', label='Map-derived smooth route')
    ax.plot(*path.start_position, 'go', label='Start / finish')
    ax.set(xlabel='World x [m]', ylabel='World y [m]', title=f'Ground-truth loop plan: {path.total_length:.1f} m', aspect='equal')
    ax.legend()
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    grid, start = map_from_scene(args.scene)
    path = PlannedLoopReference(grid, start)
    plot_plan(path, args.output)
    print(f'Planned {path.total_length:.2f} m; minimum centre clearance {grid.clearances(path.sample_xy).min():.3f} m')
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

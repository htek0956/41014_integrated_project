"""Failure-mode checks for grid search and smooth reference generation."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from path_planning import OccupancyMap, astar, map_from_scene, PlannedLoopReference, DEFAULT_SCENE


class PlanningTests(unittest.TestCase):
    def test_decorative_walls_stay_outside_corridor(self):
        grid, _ = map_from_scene()
        ys, xs = np.nonzero(~grid.occupied)
        free = grid.world(np.column_stack((xs, ys)))
        scenery = [g for g in ET.parse(DEFAULT_SCENE).findall('./worldbody/geom')
                   if g.get('name', '').startswith('scenery_wall_')]
        self.assertGreater(len(scenery), 10)
        for geom in scenery:
            centre = np.fromstring(geom.get('pos'), sep=' ')[:2]
            size = np.fromstring(geom.get('size'), sep=' ')[:2] + grid.resolution
            self.assertFalse(np.any(np.all(abs(free-centre) <= size, axis=1)), geom.get('name'))

    def test_blocked_route_fails(self):
        occupied = np.zeros((9, 9), bool)
        occupied[:, 4] = True
        grid = OccupancyMap(occupied, (0, 0), .1, robot_radius=0, margin=0)
        with self.assertRaisesRegex(ValueError, 'No safe route'):
            astar(grid, (1, 4), (7, 4))

    def test_diagonal_cannot_cross_wall_corner(self):
        occupied = np.ones((5, 5), bool)
        occupied[1, 1] = occupied[2, 2] = False
        grid = OccupancyMap(occupied, (0, 0), .1, robot_radius=0, margin=0)
        with self.assertRaisesRegex(ValueError, 'No safe route'):
            astar(grid, (1, 1), (2, 2))

    def test_loop_clearance_seam_and_progress(self):
        grid, start = map_from_scene()
        path = PlannedLoopReference(grid, start)
        self.assertGreater(path.total_length, 50)
        samples = path.spline(np.linspace(0, path.total_length, 20000))
        self.assertGreater(grid.clearances(samples).min(), grid.robot_radius + grid.margin)
        np.testing.assert_allclose(path.spline(0), path.spline(path.total_length), atol=1e-10)
        np.testing.assert_allclose(path.spline(0, 1), path.spline(path.total_length, 1), atol=1e-10)
        path.update((*start, path.start_heading))
        self.assertLess(path.progress, .1)
        self.assertFalse(path.finished)
        # Moving through the generated route must complete, even though start
        # and finish coincide. The same start pose must not finish a fresh lap.
        for s in np.arange(0, path.total_length+.04, .05):
            ref = path.reference_at(s)
            path.update((ref['x'], ref['y'], ref['heading']))
        self.assertTrue(path.finished)


if __name__ == '__main__':
    unittest.main()

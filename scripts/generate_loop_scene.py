"""Build scenery only. The driving route is subsequently planned from its walls."""

from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET


SCENE = Path(__file__).resolve().parents[1] / 'robotis_tb3' / 'scene_loop.xml'

# Shorten the architectural layout without shrinking the robot or passage width.
LAYOUT_SCALE = 0.70


def build_scene(destination=SCENE):
    # Architectural dimensions [m], not controller waypoints. This simple closed
    # polygon defines a winding passage with alternating left/right corners.
    # Option 2: unequal straight lengths, staggered notches and an asymmetric
    # boundary. These define walls only; A* still derives all driving positions.
    corners = [(0, 0), (4, 0), (4, 6), (10, 6), (10, 2), (14, 2),
               (14, -1), (22, -1), (22, 7), (18, 7), (18, 11), (24, 11),
               (24, 18), (13, 18), (13, 15), (5, 15), (5, 19), (0, 19)]
    corners = [(x*LAYOUT_SCALE, y*LAYOUT_SCALE) for x, y in corners]
    half_width = 0.80
    root = ET.Element('mujoco', model='winding_loop_ground_truth_diagnostic')
    ET.SubElement(root, 'include', file='scene_turtlebot3_waffle_pi.xml')
    # The included robot scene has extent=1. Keep distant walls visible when
    # zooming out to inspect this much larger maze without altering physics.
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'map', zfar='500')
    world = ET.SubElement(root, 'worldbody')

    for side, offset in enumerate((half_width, -half_width)):
        boundary = []
        for i, point in enumerate(corners):
            previous, following = corners[i - 1], corners[(i + 1) % len(corners)]
            vectors = [(point[0] - previous[0], point[1] - previous[1]),
                       (following[0] - point[0], following[1] - point[1])]
            normals = [(-dy / math.hypot(dx, dy), dx / math.hypot(dx, dy))
                       for dx, dy in vectors]
            denominator = 1 + sum(a*b for a, b in zip(*normals))
            boundary.append(tuple(point[j] + offset * (normals[0][j] + normals[1][j])
                                  / denominator for j in (0, 1)))

        for i, a in enumerate(boundary):
            b = boundary[(i + 1) % len(boundary)]
            # Axis-aligned wall boxes overlap slightly at joints, closing gaps.
            size = (abs(b[0] - a[0])/2 + .05, abs(b[1] - a[1])/2 + .05, .5)
            position = ((a[0] + b[0])/2, (a[1] + b[1])/2, .5)
            ET.SubElement(world, 'geom', name=f'nav_wall_{side}_{i}', type='box',
                          pos=' '.join(map(str, position)), size=' '.join(map(str, size)),
                          rgba='0.78 0.78 0.84 1' if side == 0 else '0.80 0.75 0.65 1')

    # Decorative maze fragments live beyond the enclosed passage. Reject any
    # fragment approaching the corridor, including corners. A fixed random seed
    # makes scenery reproducible and does not prescribe the robot's route.
    def distance_to_corridor(x, y):
        nearest = float('inf')
        for i, a in enumerate(corners):
            b = corners[(i + 1) % len(corners)]
            dx, dy = b[0]-a[0], b[1]-a[1]
            t = max(0., min(1., ((x-a[0])*dx + (y-a[1])*dy)/(dx*dx+dy*dy)))
            nearest = min(nearest, math.hypot(x-a[0]-t*dx, y-a[1]-t*dy))
        return nearest

    rng = random.Random(23)
    index = 0
    for layout_x in range(-6, 30, 3):
        for layout_y in range(-6, 27, 3):
            x, y = layout_x*LAYOUT_SCALE, layout_y*LAYOUT_SCALE
            for dx, dy in ((3*LAYOUT_SCALE, 0), (0, 3*LAYOUT_SCALE)):
                if rng.random() < .40:
                    continue
                # Sampling at 0.05 m, with extra clearance exceeding the sample
                # error and wall half-thickness, keeps the entire fragment out.
                if min(distance_to_corridor(x+dx*j/60, y+dy*j/60)
                       for j in range(61)) < half_width + .45:
                    continue
                ET.SubElement(world, 'geom', name=f'scenery_wall_{index}', type='box',
                              pos=f'{x+dx/2} {y+dy/2} 0.5',
                              size=f'{dx/2+.05} {dy/2+.05} 0.5',
                              rgba='0.56 0.61 0.65 1')
                index += 1

    # A task start and initial heading are legitimate planning inputs. There are
    # no intermediate route waypoints stored in this scene or the planner.
    ET.SubElement(world, 'site', name='navigation_start', pos=f'{2*LAYOUT_SCALE} 0 0.02',
                  size='.08', rgba='0.2 0.9 0.3 1')
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='unicode')
    return destination


if __name__ == '__main__':
    print(build_scene())

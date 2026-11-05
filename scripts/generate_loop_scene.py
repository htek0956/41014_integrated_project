from pathlib import Path
import math
import xml.etree.ElementTree as ET


SCENE = Path(__file__).resolve().parents[1] / 'robotis_tb3' / 'scene_loop.xml'


LAYOUT_SCALE = 0.85


def build_scene(destination=SCENE):


    corners = [(0, 0), (3, 0), (3, 3), (6, 3), (6, 0), (10, 0),
               (10, 4), (8, 4), (8, 7), (11, 7), (11, 10), (5, 10),
               (5, 8), (0, 8)]
    corners = [(x*LAYOUT_SCALE, y*LAYOUT_SCALE) for x, y in corners]
    half_width = 0.80
    root = ET.Element('mujoco', model='winding_loop_ground_truth_diagnostic')


    base = ET.parse(destination.parent / 'scene_turtlebot3_waffle_pi.xml')
    texture = base.find("./asset/texture[@name='groundplane']")
    texture.set('rgb1', '.48 .49 .50')
    texture.set('rgb2', '.46 .47 .48')
    texture.set('builtin', 'flat')
    texture.set('markrgb', '.55 .56 .57')
    material = base.find("./asset/material[@name='groundplane']")
    material.set('texrepeat', '1 1')
    material.set('reflectance', '0')
    ET.indent(base)
    base.write(destination.parent / 'scene_maze_base.xml', encoding='unicode')
    ET.SubElement(root, 'include', file='scene_maze_base.xml')


    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'map', zfar='500')
    world = ET.SubElement(root, 'worldbody')
    marker_walls = []
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
            marker_walls.append((a, b, side))

            size = (abs(b[0] - a[0])/2 + .05, abs(b[1] - a[1])/2 + .05, .5)
            position = ((a[0] + b[0])/2, (a[1] + b[1])/2, .5)
            ET.SubElement(world, 'geom', name=f'nav_wall_{side}_{i}', type='box',
                          pos=' '.join(map(str, position)), size=' '.join(map(str, size)),
                          rgba='.56 .58 .61 1')


    ET.SubElement(world, 'site', name='navigation_start', pos=f'{1.5*LAYOUT_SCALE} 0 0.02',
                  size='.02', rgba='0 0 0 0')
    from charging_station import add_station
    add_station(root, destination, x=1.5*LAYOUT_SCALE)
    from landmark_scene import add_wall_markers
    add_wall_markers(root, destination, marker_walls)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='unicode')
    return destination


if __name__ == '__main__':
    print(build_scene())

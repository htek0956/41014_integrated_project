"""Place stationary cubes in the driving scene"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np

from path_planning import DEFAULT_SCENE, map_from_scene, PlannedLoopReference


def add_obstacle(position, size=0.3, scene=DEFAULT_SCENE, name="obstacle_cube"):
    """Place a cube at a world x and y position with its bottom on the floor

    Size is the full cube side length in metres
    Reusing the same name moves or resizes the existing cube
    """

    if size <= 0:
        raise ValueError("Cube size must be positive")

    tree = ET.parse(scene)
    world = tree.getroot().find("worldbody")

    # Replace the previous cube with this name rather than adding duplicates
    existing = world.find(f"geom[@name='{name}']")
    if existing is not None:
        world.remove(existing)

    x, y = position
    half = size / 2

    # MuJoCo box sizes are half lengths and the position is the box centre
    # A centre height of half the side length puts the bottom on the floor
    ET.SubElement(world, "geom", name=name, type="box",
                  pos=f"{x:.6f} {y:.6f} {half:.6f}",
                  size=f"{half:.6f} {half:.6f} {half:.6f}",
                  rgba="0.9 0.25 0.1 1", contype="1", conaffinity="1")

    # The obstacle name keeps it out of the wall map used by the planner
    ET.indent(tree, space="  ")
    tree.write(scene, encoding="unicode")
    print(f"Added {size:.2f} m cube at world position ({x:.3f}, {y:.3f}) in {scene}")


def randomise_room_obstacles(scene, path, seed=None):
    """Vary the room cubes once before loading a simulation scene"""

    # A fresh generator varies each run while an integer seed repeats the same layout
    # Keep this separate from the seeded LiDAR range noise
    rng = np.random.default_rng(seed)
    tree = ET.parse(scene)
    world = tree.getroot().find("worldbody")
    layout = []

    # Use only the left side of the nominal route when locating a cube baseline
    # This keeps x variation centred on the route instead of drifting from the previous run
    room_route = path.sample_xy[path.sample_xy[:, 0] < .75]

    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if not name.startswith("obstacle_room_"):
            continue

        # Preserve the saved y position and find the nominal route x at that height
        position = np.fromstring(geom.get("pos"), sep=" ")
        index = np.argmin(abs(room_route[:, 1] - position[1]))
        baseline_x = room_route[index, 0]

        # Sample full side length and sideways displacement independently in metres
        # Update height as well so every cube remains resting on the floor
        size = float(rng.uniform(.3, .4))
        x_shift = float(rng.uniform(-.2, .2))
        x = float(baseline_x + x_shift)
        y = float(position[1])
        half = round(size / 2, 6)
        geom.set("pos", f"{x:.6f} {y:.6f} {half:.6f}")
        geom.set("size", f"{half:.6f} {half:.6f} {half:.6f}")

        # Save the written values for reports so the actual layout can be reproduced
        layout.append(dict(name=name, x=round(x, 6), y=round(y, 6),
                           size=round(2 * half, 6)))
        print(f"{name}: side {size:.3f} m at x {x:.3f} m and y {y:.3f} m")

    # Change only room cube attributes and keep the wall and marker geometry untouched
    tree.write(scene, encoding="unicode")
    return layout


def add_route_obstacles(near_points, size=0.6, scene=DEFAULT_SCENE):
    """Place room cubes on the planned route near supplied world positions"""

    # Rebuild the reference from the current walls because widening the room changes the route
    # This is scene preparation and does not start a robot simulation
    grid, start = map_from_scene(scene)
    path = PlannedLoopReference(grid, start)

    # Replace only cubes owned by this script so rerunning it cannot leave the old small cube behind
    tree = ET.parse(scene)
    world = tree.getroot().find("worldbody")
    for geom in list(world.findall("geom")):
        name = geom.get("name", "")
        if name == "obstacle_cube" or name.startswith("obstacle_room_"):
            world.remove(geom)
    tree.write(scene, encoding="unicode")

    # Each hint picks a nearby route sample so both cubes obstruct the actual reference
    # World positions and distances are in metres and each cube gets its own name
    for number, near in enumerate(near_points, start=1):
        index = np.argmin(np.linalg.norm(path.sample_xy - np.asarray(near), axis=1))
        position = path.sample_xy[index]
        add_obstacle(position, size, scene, name=f"obstacle_room_{number}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", nargs=2, type=float, metavar=("X", "Y"),
                        help="World position for one cube otherwise place both room cubes")
    parser.add_argument("--size", type=float, default=0.6,
                        help="Full cube side length in metres")
    args = parser.parse_args()

    if args.position is None:
        # Share the room positions with the generator rather than keeping a second layout here
        from generate_loop_scene import ROOM_OBSTACLE_HINTS
        add_route_obstacles(ROOM_OBSTACLE_HINTS, args.size)
    else:
        add_obstacle(args.position, args.size)

"""Repeatable ground-truth navigation experiment; checks actual wall contacts.

Run from the project root. Reports and plots go to a new indexed folder so
previous experiment results are preserved. No controller gains are changed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from controller import Controller, tracking_errors
from path_planning import map_from_scene, PlannedLoopReference, DEFAULT_SCENE, plot_plan
from robot import Robot


def validate(offset=0., heading_offset=0.):
    grid, start = map_from_scene()
    path = PlannedLoopReference(grid, start)
    controller = Controller(.6, .01, verbose=False)
    robot = Robot(DEFAULT_SCENE, max_yaw_rate=2, show_viewer=False)
    poses, errors, times = [], [], []
    wall_steps, limited_samples = 0, 0
    min_clearance = float('inf')
    next_control, stop_time, command = 0., None, (0., 0.)
    wall_ids = {robot.model.geom(i).id for i in range(robot.model.ngeom)
                if (robot.model.geom(i).name or '').startswith(('nav_wall_', 'scenery_wall_'))}
    try:
        normal = np.array([-np.sin(path.start_heading), np.cos(path.start_heading)])
        robot.reset((*(start + offset*normal), path.start_heading + heading_offset))
        while robot.time < 5*path.total_length/.6 + 20:
            if robot.time + 1e-10 >= next_control:
                pose = robot.get_truth()
                reference = path.update(pose, robot.get_truth_velocity())
                command = controller.update(pose, reference, .01)
                if path.finished and stop_time is None:
                    stop_time = robot.time
                poses.append(pose)
                errors.append(tracking_errors(pose, reference))
                times.append(robot.time)
                min_clearance = min(min_clearance, float(grid.clearances(np.array([pose[:2]]))[0]))
                next_control += .01
            limited = robot.apply_body_command(command, robot.physics_dt)
            limited_samples += int(not np.allclose(command, limited, atol=1e-6))
            robot.step()
            wall_steps += int(any(int(c.geom1) in wall_ids or int(c.geom2) in wall_ids
                                  for c in robot.data.contact))
            if stop_time is not None and robot.time-stop_time >= 2:
                break
        final_speed, final_yaw = robot.get_truth_velocity()
        errors = np.asarray(errors)
        metrics = dict(completed=path.finished, duration_s=robot.time,
                       initial_lateral_offset_m=offset, initial_heading_offset_rad=heading_offset,
                       path_length_m=path.total_length,
                       endpoint_error_m=float(np.linalg.norm(np.asarray(robot.get_truth()[:2])-start)),
                       cross_track_rms_m=float(np.sqrt(np.mean(errors[:, 0]**2))),
                       cross_track_max_m=float(np.max(abs(errors[:, 0]))),
                       min_centre_clearance_m=min_clearance,
                       min_conservative_body_clearance_m=min_clearance-grid.robot_radius,
                       wall_contact_physics_steps=wall_steps,
                       final_forward_speed_m_s=final_speed, final_yaw_rate_rad_s=final_yaw,
                       limited_physics_steps=limited_samples)
    finally:
        robot.close()
    return path, np.asarray(poses), np.asarray(times), metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset', type=float, default=0., help='Initial lateral offset [m]')
    parser.add_argument('--heading-offset', type=float, default=0., help='Initial heading offset [rad]')
    args = parser.parse_args()
    path, poses, times, metrics = validate(args.offset, args.heading_offset)
    results = ROOT / 'figures' / 'loop_navigation'
    results.mkdir(exist_ok=True)
    index = 1
    while (results / f'run_{index:03d}').exists():
        index += 1
    folder = results / f'run_{index:03d}'
    folder.mkdir()
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    np.savez_compressed(folder / 'trajectory.npz', time=times, pose=poses,
                        reference=path.sample_xy)
    fig = plot_plan(path)
    fig.axes[0].plot(poses[:, 0], poses[:, 1], '--', color='tab:orange', label='Actual robot trajectory')
    fig.axes[0].legend()
    fig.savefig(folder / 'tracking.png', dpi=150, bbox_inches='tight')
    print(json.dumps(metrics, indent=2))
    if not metrics['completed'] or metrics['wall_contact_physics_steps'] or metrics['min_conservative_body_clearance_m'] <= 0:
        raise SystemExit('FAILED: lap incomplete or wall clearance violated.')

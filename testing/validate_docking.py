import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import main
from robot import Robot
from visual_docking import StationDetector, VisualDocking


if __name__ == '__main__':

    dock = VisualDocking()
    dock.update(None, 0., 0.)
    assert dock.update(None, 0., 0.)[0] == 0
    main.LOG_RESULTS = True
    result = main.main(show_viewer=False)
    metrics = {name: result[name] for name in ('lap_completed', 'docked', 'dock_mode',
               'duration', 'docking_duration_s', 'endpoint_error', 'obstacle_contact_steps',
               'final_pose', 'final_body_velocity')}
    metrics['final_pose'] = list(metrics['final_pose'])
    metrics['final_body_velocity'] = list(metrics['final_body_velocity'])
    parent = ROOT / 'figures' / 'docking'
    parent.mkdir(exist_ok=True)
    index = 1
    while (parent / f'run_{index:03d}').exists():
        index += 1
    folder = parent / f'run_{index:03d}'
    folder.mkdir()
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    np.savez_compressed(folder / 'trajectory.npz', time=result['time'],
                        pose=result['pose'], route=result['reference_path'])

    robot = Robot(ROOT / 'robotis_tb3/scene_loop.xml', max_yaw_rate=2, show_viewer=False)
    try:
        robot.reset(result['final_pose'])
        image = result['docking_camera_rgb']
        observation = StationDetector(robot.model.camera('robot_camera').fovy[0]).observe(image, 0)
        cv2.imwrite(str(folder / 'docked_camera.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [result['final_pose'][0], -.45, .15]
        camera.distance, camera.azimuth, camera.elevation = 1.6, -90, -40
        robot.renderer.update_scene(robot.data, camera=camera)
        cv2.imwrite(str(folder / 'docked_station.png'),
                    cv2.cvtColor(robot.renderer.render(), cv2.COLOR_RGB2BGR))
    finally:
        robot.close()
    print(json.dumps(metrics, indent=2))
    assert metrics['docked'] and metrics['lap_completed'], 'Docking did not complete.'


    assert result['settings']['unwanted_contact_steps'] == 0, 'Robot contacted a wall or cube.'
    assert metrics['endpoint_error'] < .04, 'Station-relative position error too large.'
    assert abs(metrics['final_body_velocity'][0]) < .015, 'Robot did not stop.'
    assert observation is not None, 'Station marker is not visible from final pose.'

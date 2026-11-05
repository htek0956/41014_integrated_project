import json
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def add_wall_markers(root, scene_path, walls):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    folder = scene_path.parent.parent / 'assets' / 'localisation_markers'
    folder.mkdir(parents=True, exist_ok=True)
    assets = ET.SubElement(root, 'asset')
    world = root.find('worldbody')
    dock = json.loads((folder.parent / 'charging_station/station.json').read_text())
    records = [dict(id=0, side_length_m=.20, position_world_m=dock['position_world_m'],
                    x_axis_world=dock['x_axis_world'], y_axis_world=dock['y_axis_world'],
                    normal_world=dock['normal_world'])]
    for a, b, side in walls:
        length = math.dist(a, b)
        if length < .45:
            continue
        tangent = (np.asarray(b)-a)/length
        marker_x = np.r_[(1 if side == 0 else -1)*tangent, 0.]
        normal = np.cross(marker_x, [0, 0, 1])
        for j in range(math.ceil(length/1.2)):
            xy = np.asarray(a)+(j+.5)/math.ceil(length/1.2)*(np.asarray(b)-a)
            if np.linalg.norm(xy-np.asarray(dock['position_world_m'][:2])) < .45:
                continue
            marker_id = len(records)
            if marker_id >= 250:
                raise ValueError('DICT_4X4_250 capacity exceeded.')
            if not cv2.imwrite(str(folder / f'{marker_id}.png'), cv2.aruco.generateImageMarker(dictionary, marker_id, 300)):
                raise OSError('Could not save marker image.')
            ET.SubElement(assets, 'texture', name=f'landmark_tex_{marker_id}', type='2d',
                          file=f'../assets/localisation_markers/{marker_id}.png')
            ET.SubElement(assets, 'material', name=f'landmark_mat_{marker_id}',
                          texture=f'landmark_tex_{marker_id}', texuniform='false',
                          texrepeat='1 1', specular='0', shininess='0')
            base = np.r_[xy, .20]
            for name, offset, size, attributes in (
                ('backing', .052, '.14 .14 .001', dict(rgba='1 1 1 1')),
                ('face', .055, '.10 .10 .0005', dict(material=f'landmark_mat_{marker_id}')),
            ):
                ET.SubElement(world, 'geom', name=f'landmark_{marker_id}_{name}', type='box',
                              pos=' '.join(map(str, base+offset*normal)),
                              xyaxes=' '.join(map(str, (*marker_x, 0, 0, 1))), size=size,
                              contype='0', conaffinity='0', **attributes)
            records.append(dict(id=marker_id, side_length_m=.20,
                                position_world_m=(base+.0555*normal).tolist(),
                                x_axis_world=marker_x.tolist(), y_axis_world=[0, 0, 1],
                                normal_world=normal.tolist()))
    for record in records:
        world_tag = np.eye(4)
        world_tag[:3, :3] = np.column_stack((record['x_axis_world'], record['y_axis_world'], record['normal_world']))
        world_tag[:3, 3] = record['position_world_m']
        record['transform_world_tag'] = world_tag.tolist()
    metadata = dict(dictionary='DICT_4X4_250', frame='world: x/y/z metres, right-handed',
                    transform_convention='T_world_tag maps tag coordinates to world; tag x right, y up, z out',
                    markers=records)
    scene_path.with_suffix('.landmarks.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

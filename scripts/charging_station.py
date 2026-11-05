import json
import xml.etree.ElementTree as ET

import cv2


def add_station(root, scene_path, x):
    folder = scene_path.parent.parent / 'assets' / 'charging_station'
    folder.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker_file = folder / 'dock_0.png'
    if not cv2.imwrite(str(marker_file), cv2.aruco.generateImageMarker(dictionary, 0, 400)):
        raise OSError(f'Could not write {marker_file}')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='dock_texture', type='2d', file='../assets/charging_station/dock_0.png')
    ET.SubElement(assets, 'material', name='dock_marker_material', texture='dock_texture',
                  texuniform='false', texrepeat='1 1', rgba='1 1 1 1', specular='0', shininess='0')
    world = root.find('worldbody')

    def box(name, pos, size, rgba, **kwargs):
        return ET.SubElement(world, 'geom', name=name, type='box', pos=' '.join(map(str, pos)),
                             size=' '.join(map(str, size)), rgba=rgba, **kwargs)


    box('dock_housing', (x, -.73, .24), (.30, .04, .24), '.10 .13 .17 1')
    box('dock_foot', (x, -.70, .025), (.34, .07, .025), '.12 .16 .20 1')


    for offset in (-.055, .055):
        box(f'dock_contact_{offset}', (x+offset, -.5115, .07),
            (.035, .1725, .025), '.72 .73 .74 1')
        box(f'dock_contact_tip_{offset}', (x+offset, -.3545, .0955),
            (.035, .0155, .0005), '.85 .65 .20 1', contype='0', conaffinity='0')
    box('dock_status_light', (x, -.684, .46), (.045, .004, .012), '.15 .9 .35 1', contype='0', conaffinity='0')


    box('dock_marker_backing', (x, -.680, .20), (.14, .14, .001), '1 1 1 1',
        xyaxes='-1 0 0 0 0 1', contype='0', conaffinity='0')
    box('dock_marker', (x, -.677, .20), (.10, .10, .0005), '1 1 1 1',
        xyaxes='-1 0 0 0 0 1', material='dock_marker_material', contype='0', conaffinity='0')
    ET.SubElement(world, 'site', name='dock_target', pos=f'{x} -.2625 0.02',
                  size='.02', rgba='0 0 0 0')
    metadata = dict(dictionary='DICT_4X4_250', marker_id=0, marker_side_length_m=.20,
                    position_world_m=[x, -.6765, .20], normal_world=[0, 1, 0],
                    x_axis_world=[-1, 0, 0], y_axis_world=[0, 0, 1],
                    desired_camera_depth_m=.314, camera_forward_offset_m=.10,
                    target_base_pose=[x, -.2625, -1.5707963267948966],
                    charging_model='passive physical contacts; camera-position stop; no battery simulation')
    (folder / 'station.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

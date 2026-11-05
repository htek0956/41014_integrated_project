import json

import cv2
import numpy as np
from scipy.optimize import least_squares


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, np.asarray(translation).ravel()
    return result


class CameraPoseEstimator:
    def __init__(self, landmark_file, intrinsics, base_from_optical):
        metadata = json.loads(landmark_file.read_text(encoding='utf-8'))
        if metadata['dictionary'] != 'DICT_4X4_250':
            raise ValueError('Expected DICT_4X4_250 landmark map.')
        self.intrinsics = np.asarray(intrinsics, dtype=float)
        self.optical_from_base = np.linalg.inv(base_from_optical)
        self.base_from_optical = np.asarray(base_from_optical)
        self.distortion = np.zeros(5)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), parameters)
        self.landmarks = {}
        for marker in metadata['markers']:
            marker_id = int(marker['id'])
            if marker_id in self.landmarks:
                raise ValueError('Duplicate landmark ID.')
            h = marker['side_length_m']/2
            local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
            rotation = np.column_stack((marker['x_axis_world'], marker['y_axis_world'], marker['normal_world']))
            world_from_tag = transform(rotation, marker['position_world_m'])
            self.landmarks[marker_id] = dict(local=local, transform=world_from_tag,
                world=local@rotation.T+world_from_tag[:3, 3])

    def _base_pose(self, parameters):
        optical_from_world = transform(cv2.Rodrigues(parameters[:3])[0], parameters[3:])
        world_from_base = np.linalg.inv(optical_from_world) @ self.optical_from_base
        return np.array([world_from_base[0, 3], world_from_base[1, 3],
                         np.arctan2(world_from_base[1, 0], world_from_base[0, 0])]), world_from_base

    def _planar_camera(self, pose):


        c, s = np.cos(pose[2]), np.sin(pose[2])
        world_base = transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]), [*pose[:2], 0.])
        optical_world = np.linalg.inv(world_base @ self.base_from_optical)
        return (cv2.Rodrigues(optical_world[:3, :3])[0].ravel(), optical_world[:3, 3], world_base)

    def estimate(self, rgb, timestamp):

        result = dict(timestamp=float(timestamp), valid=False, pose=None, covariance=None,
                      ids_detected=[], ids_used=[], ids_rejected=[], reprojection_rms_px=None,
                      reason='No known readable tags', frame='world', units=('m', 'm', 'rad'),
                      covariance_note='Approximate joint-PnP covariance; 1px noise plus 2cm/2deg model floors; validate before EKF use')
        detected, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return result
        result['ids_detected'] = list(map(int, ids.ravel()))
        observations = {}
        for marker_id, corners in zip(ids.ravel(), detected):
            marker_id = int(marker_id)
            pixels = corners.reshape(4, 2).astype(float)
            if marker_id in self.landmarks and np.min(np.linalg.norm(pixels-np.roll(pixels, 1, axis=0), axis=1)) >= 12:
                observations[marker_id] = pixels
        if not observations:
            return result
        keys = list(observations)
        world = np.vstack([self.landmarks[k]['world'] for k in keys])
        pixels = np.vstack([observations[k] for k in keys])

        def errors(parameters):
            r, t = parameters[:3], parameters[3:]
            projected = cv2.projectPoints(world, r, t, self.intrinsics, self.distortion)[0].reshape(-1, 2)
            optical = world@cv2.Rodrigues(r)[0].T+t
            rms = np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1).reshape(-1, 4), axis=1))
            rms[np.min(optical[:, 2].reshape(-1, 4), axis=1) <= .05] = np.inf
            return rms


        candidates = []
        for key in keys:
            tag = self.landmarks[key]
            solved = cv2.solvePnPGeneric(tag['local'], observations[key], self.intrinsics,
                                       self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            for r, t in zip(solved[1], solved[2]):
                optical_world = transform(cv2.Rodrigues(r)[0], t) @ np.linalg.inv(tag['transform'])
                parameters = np.r_[cv2.Rodrigues(optical_world[:3, :3])[0].ravel(), optical_world[:3, 3]]
                rms = errors(parameters)
                good = rms < 3.
                if good.any():
                    candidates.append((int(good.sum()), -float(np.mean(rms[good])), parameters, good))
        if not candidates:
            result['reason'] = 'No consistent PnP solution'
            return result
        _, _, parameters, good = max(candidates, key=lambda item: item[:2])
        if len(keys) > 1:


            good = np.ones(len(keys), dtype=bool)


        for _ in range(2):
            mask = np.repeat(good, 4)
            r, t = cv2.solvePnPRefineLM(world[mask], pixels[mask], self.intrinsics,
                                      self.distortion, parameters[:3].copy(), parameters[3:].copy())
            parameters = np.r_[r.ravel(), t.ravel()]
            revised = errors(parameters) < 3.
            if np.array_equal(revised, good):
                break
            good = revised
            if not good.any():
                result['reason'] = 'Refinement rejected every tag'
                return result
        mask = np.repeat(good, 4)


        r, t = cv2.solvePnPRefineLM(world[mask], pixels[mask], self.intrinsics,
                                  self.distortion, parameters[:3].copy(), parameters[3:].copy())
        parameters = np.r_[r.ravel(), t.ravel()]
        pose, _ = self._base_pose(parameters)


        for _ in range(3):
            mask = np.repeat(good, 4)
            def planar_residual(value):
                pr, pt, _ = self._planar_camera(value)
                return (cv2.projectPoints(world[mask], pr, pt, self.intrinsics,
                                          self.distortion)[0].reshape(-1, 2)-pixels[mask]).ravel()
            fit = least_squares(planar_residual, pose, loss='soft_l1', f_scale=1., max_nfev=80)
            pose = fit.x
            r, t, world_base = self._planar_camera(pose)
            revised = errors(np.r_[r, t]) < 3.
            if np.array_equal(revised, good):
                break
            good = revised
            if not good.any():
                result['reason'] = 'Image not consistent with flat-floor camera model'
                return result
        mask = np.repeat(good, 4)


        fit = least_squares(planar_residual, pose, max_nfev=80)
        pose = fit.x
        pose[2] = (pose[2]+np.pi) % (2*np.pi)-np.pi
        r, t, world_base = self._planar_camera(pose)
        residual = planar_residual(pose).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        if rms > 3.:
            result['reason'] = 'High reprojection error'
            return result
        information = fit.jac.T @ fit.jac
        if np.linalg.cond(information) > 1e14:
            result['reason'] = 'Ill-conditioned geometry'
            return result
        covariance = np.linalg.pinv(information) * max(1., rms**2/2)
        covariance += np.diag([.02**2, .02**2, np.deg2rad(2.)**2])
        result.update(valid=True, pose=pose, covariance=(covariance+covariance.T)/2,
                      ids_used=[key for key, accepted in zip(keys, good) if accepted],
                      ids_rejected=[key for key in result['ids_detected'] if key not in keys or not good[keys.index(key)]],
                      reprojection_rms_px=rms, reason='OK', transform_world_base=world_base,
                      pose_model='flat floor: base z=0, roll=pitch=0; fixed camera mounting')
        return result

    def annotate(self, rgb, measurement):
        annotated = rgb.copy()
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            for corners_for_id, marker_id in zip(corners, ids.ravel()):
                if int(marker_id) not in measurement['ids_used']:
                    cv2.polylines(annotated, [corners_for_id.astype(np.int32)], True, (255, 70, 70), 2)
        if measurement['valid']:
            x, y, theta = measurement['pose']
            text = f'CAMERA x={x:.3f} y={y:.3f} yaw={np.rad2deg(theta):.1f}deg | tags={len(measurement["ids_used"])}'
        else:
            text = 'CAMERA: no valid pose - '+measurement['reason']
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(annotated, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated

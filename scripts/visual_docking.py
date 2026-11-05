import cv2
import numpy as np


class StationDetector:
    def __init__(self, fovy_degrees, width=640, height=480, marker_id=0, marker_size=.20):
        focal = height / (2*np.tan(np.deg2rad(fovy_degrees)/2))
        self.intrinsics = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
        self.distortion = np.zeros(5)
        self.marker_id = marker_id
        h = marker_size/2

        self.corners = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), parameters)

    def observe(self, rgb, timestamp):
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(grey)
        if ids is None or self.marker_id not in ids:
            return None
        pixels = corners[list(ids.ravel()).index(self.marker_id)].reshape(4, 2)
        ok, rotation, translation = cv2.solvePnP(self.corners, pixels, self.intrinsics,
                                               self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        t = translation.ravel()
        normal = cv2.Rodrigues(rotation)[0][:, 2]
        projected = cv2.projectPoints(self.corners, rotation, translation,
                                     self.intrinsics, self.distortion)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
        if not np.all(np.isfinite(t)) or not .15 < t[2] < 2.5 or normal[2] >= -.3 or error > 2:
            return None
        return dict(timestamp=float(timestamp), translation=t, normal=normal,
                    reprojection_error_px=error, corners=pixels)


class VisualDocking:
    def __init__(self):
        self.mode = 'BRAKE'
        self.finished = False
        self.stable_samples = 0
        self.last_observation = None
        self.seating_samples = 0
        self.seating = False

    def update(self, observation, speed, yaw_rate):

        settled = abs(speed) < .015 and abs(yaw_rate) < .04
        if self.finished:
            return 0., 0.
        if self.mode == 'BRAKE':
            if settled:
                self.mode = 'ACQUIRE'
            return 0., 0.
        if observation is None:
            if self.seating:
                return 0., 0.


            self.mode = 'ACQUIRE'
            self.stable_samples = 0
            return 0., -.30
        self.last_observation = observation
        t, n = observation['translation'], observation['normal']
        heading_error = float(np.arctan2(n[0], -n[2]))
        bearing = float(np.arctan2(t[0], t[2]))


        if self.seating or (t[2] < .335 and abs(t[0]) < .02 and abs(heading_error) < .08):
            self.seating = True
            self.mode = 'SEAT'
            if abs(t[0]) > .035 or abs(heading_error) > .12 or t[2] < .285:
                return 0., 0.
            self.seating_samples += 1
            if self.seating_samples <= 40:
                return .015, float(np.clip(-1.5*bearing, -.08, .08))
            self.stable_samples = self.stable_samples+1 if settled else 0
            if self.stable_samples >= 10:
                self.mode, self.finished = 'DOCKED', True
            return 0., 0.
        self.mode = 'DOCK'
        if t[2] < .37:
            return (.025 if t[2] > .335 else 0.), float(np.clip(-1.5*bearing+.5*heading_error, -.15, .15))


        forward = t[2] + .10 + .414*n[2]
        left = -t[0] - .414*n[0]
        distance = float(np.hypot(forward, left))


        if distance > .06 and abs(heading_error) > .06:
            return 0., float(np.clip(1.5*heading_error, -.30, .30))
        self.stable_samples = 0
        alpha = float(np.arctan2(left, forward))
        beta = (heading_error-alpha+np.pi) % (2*np.pi)-np.pi


        if abs(alpha) > 1.2:
            return 0., float(np.clip(1.5*alpha, -.40, .40))
        forward_speed = min(.12 if distance > .08 else .025, .6*distance) * max(0., np.cos(alpha))
        turn_rate = float(np.clip(2*alpha-.6*beta, -.40, .40))
        return float(forward_speed), turn_rate

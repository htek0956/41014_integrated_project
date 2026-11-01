from pathlib import Path
from mujoco import viewer

import mujoco


class Robot:
    def __init__(self):
        scene = Path(__file__).resolve().parent.parent / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.actuators = [self.model.actuator(name).id
                          for name in ("wheel_left", "wheel_right")]
        mujoco.viewer.launch(self.model, self.data)
    def get_state(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def step(self, outputs):
        self.data.ctrl[self.actuators] = outputs
        mujoco.mj_step(self.model, self.data)

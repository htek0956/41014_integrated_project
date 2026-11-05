import mujoco
import mujoco.viewer
import time
from pathlib import Path

scene_path = Path(__file__).resolve().parent.parent / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

actuators = [model.actuator(name).id for name in ("wheel_left", "wheel_right")]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        data.ctrl[actuators] = 5.0 if data.time < 5.0 else 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


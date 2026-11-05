import time
import mujoco
import mujoco.viewer
from pathlib import Path


model_path = Path(__file__).parent / "universal_robots_ur3e" / "ur3e.xml"


model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)


mujoco.mj_resetDataKeyframe(model, data, 0)


base_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan")
slow_velocity = 0.1

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()


        data.ctrl[base_actuator_id] += slow_velocity * model.opt.timestep


        mujoco.mj_step(model, data)
        viewer.sync()


        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

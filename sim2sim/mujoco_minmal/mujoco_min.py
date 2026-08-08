import time
from pathlib import Path

import mujoco
import mujoco.viewer

script_dir = Path(__file__).resolve().parent
xml_path = script_dir / "hello.xml"
# 1. 编译并加载 XML，得到静态模型
model = mujoco.MjModel.from_xml_path(str(xml_path))

# 2. 根据模型创建动态仿真数据
data = mujoco.MjData(model)

# 3. 创建非阻塞可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 4. 推进一个物理仿真步
        mujoco.mj_step(model, data)

        # 5. 把最新状态同步到可视化窗口
        viewer.sync()

        # 6. 让仿真速度接近真实时间
        remaining_time = model.opt.timestep - (time.time() - step_start)
        if remaining_time > 0:
            time.sleep(remaining_time)
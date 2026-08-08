import argparse
from pathlib import Path

import mujoco


parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("robot.xml"))
args = parser.parse_args()

urdf_path = args.urdf.resolve()
output_path = args.output.resolve()

model = mujoco.MjModel.from_xml_path(str(urdf_path))
mujoco.mj_saveLastXML(str(output_path), model)

print(f"[INFO] URDF: {urdf_path}")
print(f"[INFO] MJCF: {output_path}")
"""Load the Unitree Go2 MuJoCo model on a flat ground.

This is the first minimal sim2sim checkpoint:
- no SDK
- no DDS
- no policy
- no PD control

It loads a fixed flat scene XML and opens MuJoCo's passive viewer.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_PATH = PROJECT_DIR / "scenes" / "go2_flat_scene.xml"
DEFAULT_GO2_HOME_QPOS = [
    0.0,
    0.0,
    0.27,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
]


def find_unitree_go2_dir(unitree_mujoco_dir: Path | None = None) -> Path:
    """Find ``unitree_mujoco/unitree_robots/go2`` without hard-coded paths."""
    candidates: list[Path] = []

    if unitree_mujoco_dir is not None:
        candidates.append(unitree_mujoco_dir)

    env_dir = os.environ.get("UNITREE_MUJOCO_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            PROJECT_DIR / "unitree_mujoco",
            PROJECT_DIR.parent / "unitree_mujoco",
            PROJECT_DIR.parent / "unitree_mujoco-main",
        ]
    )

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        go2_dir = candidate / "unitree_robots" / "go2"
        checked.append(go2_dir)
        if (go2_dir / "go2.xml").exists():
            return go2_dir

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Cannot find Unitree Go2 MJCF. Clone unitree_mujoco next to this project, "
        "set UNITREE_MUJOCO_DIR, or pass --unitree-mujoco-dir. Checked:\n"
        f"{checked_text}"
    )


def prepare_scene_for_mujoco(scene_path: Path, go2_dir: Path) -> Path:
    """Materialize the course scene beside Unitree's go2.xml.

    Unitree's go2.xml uses ``<compiler meshdir="assets">``. Loading a runtime
    scene next to ``go2.xml`` lets MuJoCo resolve meshes exactly as in Unitree's
    official scenes, while keeping the editable course scene under ``scenes/``.
    """
    go2_xml = go2_dir / "go2.xml"
    if not go2_xml.exists():
        raise FileNotFoundError(f"Cannot find Unitree Go2 XML: {go2_xml}")

    scene_text = scene_path.read_text(encoding="utf-8")
    scene_text = scene_text.replace("{{GO2_XML}}", "go2.xml")

    runtime_scene = go2_dir / "go2_flat_scene_runtime.xml"
    runtime_scene.write_text(scene_text, encoding="utf-8")
    return runtime_scene


def print_model_summary(model: mujoco.MjModel) -> None:
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("[INFO] Loaded model:")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("[INFO] Joints:")
    for index, name in enumerate(joint_names):
        print(f"  {index:02d}: {name}")
    print("[INFO] Actuators:")
    for index, name in enumerate(actuator_names):
        print(f"  {index:02d}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Unitree Go2 in a flat MuJoCo scene.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to the MuJoCo scene XML.",
    )
    parser.add_argument(
        "--unitree-mujoco-dir",
        type=Path,
        default=None,
        help="Path to the cloned unitree_mujoco repository. Defaults to auto-discovery.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Load the model and step headlessly.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to keep the viewer open.")
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Cannot find scene XML: {scene_path}")

    go2_dir = find_unitree_go2_dir(args.unitree_mujoco_dir)
    runtime_scene_path = prepare_scene_for_mujoco(scene_path, go2_dir)
    print(f"[INFO] Loading scene: {scene_path}")
    print(f"[INFO] Unitree Go2 dir: {go2_dir}")
    print(f"[INFO] Runtime scene: {runtime_scene_path}")

    model = mujoco.MjModel.from_xml_path(str(runtime_scene_path))
    data = mujoco.MjData(model)

    # Start from Unitree's home keyframe if it exists; otherwise write the known home qpos.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        print("[INFO] Reset with keyframe: home")
    elif model.nq == len(DEFAULT_GO2_HOME_QPOS):
        data.qpos[:] = DEFAULT_GO2_HOME_QPOS
        mujoco.mj_forward(model, data)
        print("[INFO] Reset with built-in Go2 home qpos")
    else:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        print("[INFO] Reset with MuJoCo default state")

    print_model_summary(model)

    if args.no_viewer:
        steps = int(args.duration / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        print(f"[INFO] Headless simulation finished at t={data.time:.3f}s")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < args.duration:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

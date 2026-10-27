"""Record an oracle pick-and-place episode from the fixed camera and a wide view (mp4)."""
import argparse

import imageio
import mujoco
import numpy as np

from aura3d.controllers.pick_place import pick_and_place
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="ur10e")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
env = WorkcellEnv(WorkcellCfg(robot=a.robot))
env.reset(a.seed)
r = mujoco.Renderer(env.model, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [*env.meta["table_center"], 0.2]
cam.distance, cam.azimuth, cam.elevation = (2.6 if "fanuc" in a.robot else 2.0), 150, -28
frames, every = [], round((1 / 30) / env.model.opt.timestep)
count = [0]
orig = env.step


def step(n=1):
    for _ in range(n):
        orig()
        count[0] += 1
        if count[0] % every == 0:
            r.update_scene(env.data, camera=cam)
            frames.append(r.render().copy())


env.step = step
for i in range(3):
    pick_and_place(env, env.grasp_world(i, env.gt_part_pose(i)), ("bin_a", "bin_b")[i % 2])
imageio.mimwrite(a.out, frames, fps=30, quality=7)
print("wrote", a.out, len(frames), "frames", "placed:", [env.part_in_bin(i, ("bin_a", "bin_b")[i % 2]) for i in range(3)])

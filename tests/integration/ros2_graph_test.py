"""ROS 2 integration test (run inside the ROS 2 Jazzy container; not collected by the local pytest run).

Spins SimBridge + Perception + CalibMonitor in one executor. Checks:
  1. part poses arrive (3 parts) with nominal extrinsics, and the monitor reports no alarm;
  2. after setting the bridge parameter drift_rot_deg = 1.0, the monitor raises the alarm;
  3. records frame -> node latency and messages per topic.
Writes results/ros2/graph_test.json; exits non-zero on failure.
"""
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String

from aura3d.calibration.monitor import DriftMonitor
from aura3d.cameras.noise import PRESETS
from aura3d.ros2.nodes import CalibMonitor, Perception, SimBridge
from aura3d.simulation.env import WorkcellEnv
from aura3d.simulation.workcell import WorkcellCfg

rclpy.init()
env = WorkcellEnv(WorkcellCfg(robot="ur10e"), noise=PRESETS["nominal"])
frames = []
for s in range(20):
    env.reset(s)
    frames.append(env.render("fixed"))
mon = DriftMonitor.commission(frames, env.meta)
T_believed = env.meta["cameras"]["fixed"]["T_world_cam"]
bridge, perc, cm = SimBridge(env, seed=1000), Perception(env.meta, T_believed), CalibMonitor(env.meta, T_believed, mon)
probe = rclpy.create_node("aura_probe")
got = {"classes": [], "alarm": []}
probe.create_subscription(String, "/aura/part_classes", lambda m: got["classes"].append(json.loads(m.data)), 10)
probe.create_subscription(Bool, "/aura/calib_alarm", lambda m: got["alarm"].append(bool(m.data)), 10)
ex = SingleThreadedExecutor()
for n in (bridge, perc, cm, probe):
    ex.add_node(n)


def spin(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.spin_once(timeout_sec=0.05)


spin(6.0)
nominal = {"classes": list(got["classes"]), "alarm": list(got["alarm"])}
got["alarm"].clear()
bridge.set_parameters([Parameter("drift_rot_deg", Parameter.Type.DOUBLE, 1.0)])
spin(6.0)
drifted = list(got["alarm"])
res = {"nominal_frames": len(nominal["alarm"]), "nominal_alarms": sum(nominal["alarm"]),
       "nominal_n_parts_last": len(nominal["classes"][-1]) if nominal["classes"] else 0,
       "drift_frames": len(drifted), "drift_alarms": sum(drifted),
       "perception_latency_ms_p50": 1000 * sorted(perc.latency_s)[len(perc.latency_s) // 2] if perc.latency_s else None}
ok = (res["nominal_frames"] > 0 and res["nominal_alarms"] == 0 and res["nominal_n_parts_last"] == 3
      and res["drift_frames"] > 0 and res["drift_alarms"] >= max(1, res["drift_frames"] - 2))
res["pass"] = bool(ok)
Path("results/ros2").mkdir(parents=True, exist_ok=True)
Path("results/ros2/graph_test.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
rclpy.shutdown()
sys.exit(0 if ok else 1)

# ROS 2 integration

ROS 2 is an integration layer over the same Python modules. It is not a substitute for the direct-Python experiments,
and nothing here claims compatibility with any vendor controller or with Palladyne IQ.

**Runtime:** `ros:jazzy-ros-base` (Ubuntu 24.04, Python 3.12) via Apptainer on an NU Explorer CPU node
(`scripts/cluster/ros2_test.sbatch`). Python deps are in a user base on scratch (the image has no python3-venv).
MuJoCo renders with OSMesa (conda mesalib).

**Nodes** (`src/aura3d/ros2/nodes.py`), standard messages only:
| Node | Publishes | Subscribes |
|---|---|---|
| `aura_sim_bridge` | `/aura/fixed/{rgb,depth,camera_info}` (sensor-data QoS), `/joint_states`, TF `world → aura/fixed_optical` (believed extrinsics); parameter `drift_rot_deg` injects a camera bump at runtime | – |
| `aura_perception` | `/aura/part_poses` (PoseArray, world), `/aura/part_classes` (String, JSON aligned with poses) | rgb + depth + camera_info, exact-stamp pairing |
| `aura_calib_monitor` | `/aura/calib_residual_mm` (Float32), `/aura/calib_alarm` (Bool) | same |

**Integration test** (`tests/integration/ros2_graph_test.py`, SLURM 11413626): **PASS**.
- Nominal: 20 synced frames, 0 alarms, 3 parts on `/aura/part_poses`.
- After `drift_rot_deg := 1.0`: 4 frames, 2 alarms. The first frames after the change were likely rendered before it,
  so this measures alarm latency, not missed detection. It was not separately measured.
- Stamp → callback latency p50 3.2 ms. This is transport only, not perception compute.

**Not done:** custom message types (class labels travel as JSON), lifecycle nodes, MoveIt 2, a command interface to
the robot (picking still runs in-process), QoS stress tests, multi-machine.

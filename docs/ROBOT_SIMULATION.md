# Robot simulation

## Embodiments (configs/robots.yaml)
| | UR10e | FANUC M-710iC/50 |
|---|---|---|
| Source | MuJoCo Menagerie `f3b6013` (BSD-3) | FANUC-CORPORATION/fanuc_description `fb40c98` (Apache-2.0) |
| Import | MJCF, Menagerie actuators replaced | xacro → URDF (`scripts/import_fanuc_m710.py`) → MjSpec |
| Kinematics / limits / inertials | Menagerie | **FANUC's file, unchanged** (a1 0.15, a2 0.87, a3 0.17, d4 1.016, d6 0.175 m; link masses 130/201/91/95/33/9 kg) |
| Visual meshes | Menagerie OBJ | FANUC collision STL (MuJoCo cannot read the .dae visuals) |
| Servo gains | ours (kp 3000/2000/500) | ours (kp 40000/30000/4000…); **not FANUC servo parameters** |
| Gravity compensation | on | on |

Both robots carry the same parallel-jaw gripper (ours: two slide fingers, one servo, equality coupling, 60 N limit,
max jaw 0.11 m) so perception and grasp logic are embodiment-independent.

## Issues found and fixed while building (all reproducible)
1. **FANUC J1 pinned.** Base and J1 convex hulls overlap. MuJoCo's parent-child contact filter is disabled for bodies
   welded to the world, so the base–J1 contact was active. Fix: explicit contact excludes for every robot parent-child pair.
2. **Servo lag at grasp.** The TCP was still 8–15 mm above the target when the jaws closed. Fix: `move_to` waits
   (bounded, 0.6 s) until every joint is within 2 mrad.
3. **Pad on tray wall.** A fully open gripper (0.11 m) landed on the tray wall for parts near it. Fix: pre-shape to
   grasp width + 25 mm, and keep parts ≥ 55 mm from the walls.
4. **UR10e IK local minima / bad branch.** Single-seed DLS failed; random restarts then found an elbow-under-mount
   solution the arm could not track. Fix: deterministic restarts plus IK branch bounds in config (shoulder-lift ∈ [−π, 0],
   elbow ∈ [0, π]; pan keeps its full ±2π because the Menagerie base is rotated 180°).

## Evidence
EXP-M2-001 (oracle poses, seeds 1000–1099, 3 parts each): **UR10e 300/300, FANUC 300/300** placed, Wilson 95 %
[0.987, 1.0]. Mean cycle time per pick 7.9 s (UR10e) and 7.4 s (FANUC) of simulated time; these are our motion
timings, not either vendor's cycle times. Tests: `tests/simulation/test_workcell.py` (14 cases × 2 robots).

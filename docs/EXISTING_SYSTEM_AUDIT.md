# Existing system audit (2026-10-27)

## BELIEF-3D (`~/belief-3d`, public repo parammadan/belief-3d, MIT, same author)
- HEAD `7976b64` ("paper: qualify replay claim"). ~8.7k lines of Python.
- **Tests re-run today:** `pytest -q` → **221 passed, 1 skipped in 15.7 s** (local venv: Python 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4).
- Robot: Franka Panda only (Menagerie `f3b6013`). No UR10e, no FANUC, no ROS 2, no torch in the local venv.
- Results it reports (M3–M8: neural pose + SDF refinement, belief tracking, active perception, self-calibration,
  closed-loop pick) are taken from its `PROJECT_STATE.md` and `EXPERIMENT_REGISTRY.md`. **They were not re-run
  for this audit**, and AURA-3D does not cite them as its own results.
- The M3 neural-pose checkpoint lives only on the cluster (`/scratch/madan.pa/belief3d`) and was not re-verified.

### Reused in AURA-3D (each file marks its provenance in its header)
| BELIEF-3D file | AURA-3D file | Change |
|---|---|---|
| `perception/geometry/se3.py` | `src/aura3d/geometry/se3.py` | unchanged |
| `perception/geometry/camera.py` | `src/aura3d/geometry/camera.py` | unchanged |
| `tests/unit/test_se3.py`, `test_camera.py` | `tests/unit/` | import paths only |
| `sim/robot/panda.py` (IK, servo) | `src/aura3d/robots/manipulator.py` | generalised to any robot from config; IK restarts + branch bounds; settle check |
| `sim/cameras/noise.py` (structure) | `src/aura3d/cameras/noise.py` | replaced placeholder coefficients with a stereo-triangulation model |
| `manipulation/grasping/executor.py` (sequence) | `src/aura3d/controllers/pick_place.py` | adds place-in-bin, pre-shaped jaws, blocked-motion result |

Not reused yet (candidates): `perception/calibration/self_calib.py`, belief tracker, neural proposer (needs torch on
GPU), C++ kernels.

## Other prior work
- `~/Desktop/exp/palladyne recruiter/research/*.md`: planning documents (no code). The "FWPA" design there is a plan;
  no FWPA code existed before this repo.

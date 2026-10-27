# AURA-3D: 5-minute technical presentation

*Simulation only. Every number comes from `results/` and the pre-registered `EXPERIMENT_REGISTRY.md`.*

---

## 1. The problem (30 s)
A vision-guided pick cell is calibrated once and then trusted. When the camera is bumped, everything downstream is
wrong, and the pose estimator still reports high confidence. In our cell a **0.25° bump drops sorting success from
50/50 to 29/50**, and a 0.5° bump drops it to 7/50.

## 2. What I built (45 s)
- MuJoCo workcell with **FANUC M-710iC/50 (FANUC's official description)** and a UR10e, a shared gripper, an
  OAK-D-Pro-like stereo camera, and three reference spheres.
- Classical depth perception, then a reliability loop:
  **drift monitor → validated recalibration → pose-risk gate → bounded recovery**.
- A learned stack around it: Mask R-CNN fine-tuning, Meta's JEPA world model, an own latent world model.
- ROS 2 Jazzy nodes and an integration test. Public repo: github.com/parammadan/aura3d

## 3. Headline result: drift (60 s)
| Camera bump | Perception only | + monitor & validated recal |
|---|---|---|
| none | 50/50 | 50/50 (0 false alarms) |
| 0.5° | 7/50 | **42/50** (p = 6e-11) |
| 10 mm | 6/50 | **30/50** |
Wrong picks: **251 → 0** at 0.5°. What's left are *deferrals*: the system refuses a correction it cannot verify and
asks for an operator instead of acting.

## 4. Two engineering findings (60 s)
1. **Flat fixtures make translation unobservable.** A 10 mm bump is still 7.5 mm off after recalibration. Three reference
   spheres bring it to 1.2 mm. The information-matrix condition number did *not* flag bad fits, so I accept a fit only
   if it predicts a held-out sphere within 2 mm.
2. **Most drift failures are phantom parts, not pose error.** Measured grasp tolerance is 10–15 mm, but a tilted
   camera turns patches of table into "parts". A learned risk model catches them (AUROC 0.97 vs 0.77 for the
   estimator's own score). Its conformal bounds hold in distribution (0.89 of 0.90) and **fail under shift (0.71)**.

## 4b. Learned vision earns its place in clutter (20 s)
Touching parts merge in depth. Fine-tuned Mask R-CNN lifts first-frame localisation from 60 % to 99 % and sorting from
30/100 to 53/100 (p = 1.5e-5), close to the 58/100 oracle ceiling, where the gripper becomes the limit.
Picking the part with the most jaw clearance first fixes that: **50 → 88/100**, equal to oracle poses with the same rule.

## 4c. A risk gate is only as good as the perception behind it (20 s)
With classical perception in clutter the gate rightly refuses most estimates and stalls. With fine-tuned Mask R-CNN it
removes the bad ones: wrong picks −49 %, success 21 → **31/50** (p = 0.03), with only 2 deferrals.

## 5. World models: what worked and what didn't (60 s)
- Reproduced Meta JEPA-WM on its own task: 5/12 vs the paper's 41.6 %.
- **Zero-shot on our robot it is worse than predicting "nothing moves"** (2.7×). Fine-tuning only the predictor
  brings it to **0.67×** (15 epochs, V100, 67 min).
- My own latent model predicts well (0.43×), but **as a grasp verifier it made picking worse: 99 → 88/100**. Good
  prediction is not good decision-making, so world models get authority over actions only after their outcome
  ranking is validated (pre-set gate AUROC ≥ 0.80; not passed).
- Diagnostic: grasp success is visible in the *real* final frame (AUROC 0.99) but not in JEPA's *predicted* one (0.72).
  The models learn the scene's motion, not the contact outcome.

## 6. Why this matters for a product team (30 s)
- The value is in **turning silent wrong actions into explicit, explainable deferrals**, measured with paired tests.
- Everything that worked is classical + small learned models under permissive licenses. JEPA-WMs is CC BY-NC.
- Next: real sensor data (bump statistics, metal parts), gradual drift, robot self-observation for recalibration,
  and patch-level world-model outcome probes.

## Likely questions
- *Is this real-robot ready?* No. Simulation, assumed noise, primitive parts. The protocol transfers; the numbers don't.
- *Why trust the stats?* Seeds are disjoint per split, cells are paired, analysis is registered before the test run,
  and nulls and negatives are reported (H3, M8).
- *FANUC specifics?* Same code with only a config change: 0.5° bump 6/50 → 42/50 on the M-710iC/50 (p = 3e-11).

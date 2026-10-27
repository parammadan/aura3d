# EXP-M8-001 notes (partial: rot 0° done, rot 1.0° running on the cluster)

**Control (0° drift, 100 seeds):** geometry G 99/100; WM verification 88/100 (1 deferral); oracle 100/100.
WM made 11 wrong actions vs G's 1; exact McNemar WM vs G p = 0.00098. **The world-model verifier hurts at zero drift.**

Why: the fallback model predicts frames well (h = 6 latent error 0.43× copy) but its outcome head ranks outcomes
poorly (test AUROC 0.58). So argmax P(success) over the ±16 mm grid moves grasps *off* a correct estimate.
**Good latent prediction did not translate into good action selection**, the failure the master prompt warns about
(rules 6–7). Any verifier needs an outcome model validated for ranking *before* it gets authority over actions.

# EXP-M8-002 (gate result): NOT deployed
Retrained with class-balanced BCE and outcome weight 1.0 (SLURM 11413813). Outcome AUROC val **0.669**, test 0.54, below the
pre-registered 0.80 gate, so the verifier gets no authority over actions. Dynamics stayed good (h = 6: 0.456× copy val,
0.451× test; shuffled actions 0.63–0.64×).

Diagnosis (hypothesis, untested): grasp success depends on millimetre offsets between the jaws and the part. A global
image embedding (DINOv2 CLS + mean of patch tokens from a 224 px wide view) averages away exactly that detail. Next
candidates: gripper-centred crops, patch-level latents (as JEPA-WM keeps), or depth features.

# EXP-M8-001 at 1.0° drift (complete)
G 21/100, WM 16/100 (3 deferred), oracle 94/100; wrong actions G 79, WM 81; McNemar WM vs G p = 0.125.
The 1° bump puts the estimate ≈ 17 mm off (beyond tolerance). A verifier with a weak outcome head (AUROC 0.58) cannot
find the correct candidate even though it lies inside the ±16 mm grid. **H4 is not supported with this world model.**

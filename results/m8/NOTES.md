# EXP-M8-001 notes (partial: rot 0° done, rot 1.0° running on the cluster)

**Control (0° drift, 100 seeds):** geometry G 99/100; WM verification 88/100 (1 deferral); oracle 100/100.
WM made 11 wrong actions vs G's 1; exact McNemar WM vs G p = 0.00098. **The world-model verifier hurts at zero drift.**

Why: the fallback model predicts frames well (h = 6 latent error 0.43× copy) but its outcome head ranks outcomes
poorly (test AUROC 0.58). So argmax P(success) over the ±16 mm grid moves grasps *off* a correct estimate.
**Good latent prediction did not translate into good action selection**, the failure the master prompt warns about
(rules 6–7). Any verifier needs an outcome model validated for ranking *before* it gets authority over actions.

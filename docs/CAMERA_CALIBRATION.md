# Camera geometry, drift and recalibration

## Model
Pinhole (OpenCV convention), 640×400, 80° HFOV (the OAK-D Pro FF stereo FOV, native 1280×800 halved). Distortion is
zero in simulation. Depth noise: σ_z = z²·σ_d / (f_native·b), with b = 75 mm and f_native ≈ 763 px. σ_d = 0.1 px
("nominal") or 0.3 px ("degraded") is **assumed**, not fitted to a real unit.

## Fixture residual (drift signal)
Static-scene points (table footprint, tray interior excluded) are mapped with the believed extrinsics and scored by
signed distance to the workcell CAD (table, tray/bin walls, optional reference spheres on posts).

Measured (UR10e, nominal noise, dev seeds): no drift gives a median |r| of 0.98 mm (sd 0.015 across scenes). A 0.25° bump
gives at least 1.16 mm, and 2° gives ~6–12 mm. With degraded noise the floor rises to 2.83 mm, so the threshold must
come from commissioning, not a constant.

## Recalibration
Robust (Huber) least squares over a camera-frame SE(3) increment against the fixture SDF.

| Drift | Without reference spheres: error after fit | With 3 spheres |
|---|---|---|
| 1°, 0 mm | 0.06°, 1.0 mm | 0.06°, 0.96 mm |
| 0°, 10 mm | 0.81°, **7.5 mm** | 0.07°, 1.2 mm |
| 3°, 20 mm | 0.90°, **7.8 mm** | 0.08°, 1.4 mm |

The flat table plus 24–40 mm walls leaves in-plane translation weakly observable. The condition number of JᵀJ (~400–800)
did **not** separate good from bad fits, so it is not used for acceptance.

**Acceptance = leave-one-sphere-out validation:** refit without sphere k, then predict sphere k; accept only if at least 2
spheres are visible and each is within 2 mm. Without spheres the status is "unvalidated" and the fit is not applied.
The 3°/20 mm case fails validation (one held-out sphere off by 4.7 mm) and is refused, which is the intended
conservative behaviour.

## Monitor (B3cal)
Commission μ0 and σ0 on 20 drift-free dev frames; alarm at r > μ0 + 6σ0 (UR10e nominal: 0.982 + 6·0.017 = 1.085 mm).
On an alarm: recalibrate; apply only if accepted, else defer (operator). Results: EXP-M5-001 (`results/m5/ANALYSIS.md`).

## Limitations
Drift is a rigid camera-frame bump applied before the episode. Gradual drift, intrinsics error and timing skew are
implemented as parameters but not yet evaluated. Real mounts also flex with temperature, which is not modelled.

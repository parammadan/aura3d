# Perception (B1 classical baseline)

Depth → world points (believed extrinsics) → crop to tray interior, 4–90 mm above the table → image connected
components (≥ 25 px) → class from the height profile (95th-percentile height > 40 mm ⇒ pin; > 35 % of points
below 20 mm ⇒ bracket; else block) → planar pose (x, y, yaw) by Huber point-to-SDF least squares against the catalog
primitive, 4 yaw starts (1 for the pin). z, roll and pitch are fixed by the resting assumption.

Pose accuracy vs ground truth (UR10e, 30 dev scenes, 90 parts; detection and classification 90/90 in both rows):
| Noise | block xy mm (median / max) | yaw ° (median / max) | bracket xy mm | bracket yaw ° |
|---|---|---|---|---|
| nominal | 0.12 / 0.62 | 0.19 / 0.72 | 0.21 / 0.56 | 0.30 / 0.94 |
| degraded | 0.97 / 4.02 | 1.01 / 3.44 | 1.10 / 4.13 | 0.75 / 4.17 |

Known failure modes (observed, not yet fixed in B1):
- **Phantom detections:** a tilt bump plus degraded noise lifts patches of the table above the 4 mm height gate, giving
  small spurious "bracket" blobs (25–90 px).
- **Merged blobs:** touching parts form one connected component, so the pose is wrong. This motivates learned instance
  segmentation (M4).
- No RGB is used. Colour-independent by design, but blind to flat parts below 4 mm.

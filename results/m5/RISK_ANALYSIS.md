# EXP-M5-003 pose-risk results

τ (mm, from EXP-M5-002): {'block': 15.0, 'pin': 10.0, 'bracket': 12.0}

## test  (rows 1322, out of tolerance 486)

| arm | wrong-accept @20% abstain | AUROC | AUPRC | risk-coverage AUC |
|---|---|---|---|---|
| a_quality | 0.292 | 0.754 | 0.672 | 0.182 |
| b_rms | 0.292 | 0.754 | 0.672 | 0.182 |
| c_gb_error | 0.211 | 0.995 | 0.993 | 0.080 |
| d_gb_p_out | 0.211 | 0.993 | 0.987 | 0.082 |

(c) − (a) wrong-accept: -0.081, 95% CI [-0.103, -0.059]

- conformal_marginal: coverage 0.890 (target 0.90), mean bound 22.3 mm, per class {"block": 0.954, "bracket": 0.849, "pin": 0.93}
- conformal_mondrian: coverage 0.885 (target 0.90), mean bound 22.3 mm, per class {"block": 0.911, "bracket": 0.871, "pin": 0.896}

## shift  (rows 1728, out of tolerance 1536)

| arm | wrong-accept @20% abstain | AUROC | AUPRC | risk-coverage AUC |
|---|---|---|---|---|
| a_quality | 0.873 | 0.461 | 0.904 | 0.915 |
| b_rms | 0.873 | 0.461 | 0.904 | 0.915 |
| c_gb_error | 0.862 | 0.939 | 0.991 | 0.696 |
| d_gb_p_out | 0.862 | 0.934 | 0.987 | 0.703 |

(c) − (a) wrong-accept: -0.011, 95% CI [-0.017, -0.005]

- conformal_marginal: coverage 0.707 (target 0.90), mean bound 41.8 mm, per class {"block": 0.487, "bracket": 0.771, "pin": 0.34}
- conformal_mondrian: coverage 0.710 (target 0.90), mean bound 42.3 mm, per class {"block": 0.329, "bracket": 0.798, "pin": 0.289}

## Interpretation (written after the registered analysis)
- **Primary (test):** wrong-accept at 20 % abstention 0.211 (GB) vs 0.292 (quality score); difference −0.081,
  95 % scene-bootstrap CI [−0.103, −0.059]. H1 supported in-distribution. With 36.8 % of rows out of tolerance,
  0.211 is the floor any ranking can reach at 20 % abstention, so this metric saturates. Risk-coverage AUC (0.080 vs
  0.182) is more informative.
- Quality and RMS arms are identical because quality = exp(−rms / 2 mm) is monotone in rms.
- **Conformal (test):** marginal coverage 0.890 (target 0.90). The bracket is under-covered (0.849 marginal, 0.871
  Mondrian). Rows within a scene are not exchangeable, which is one plausible reason.
- **Under shift the guarantee fails, as expected:** coverage 0.71, with block 0.33–0.49 and pin 0.29–0.34. Ranking
  still works (AUROC 0.94 vs 0.46), but bounds calibrated in-distribution are not trustworthy after a shift.

## Post-hoc (NOT registered): excluding phantom / unmatched detections (e = 50 mm cap)
Most out-of-tolerance test rows (435 of 486) are phantom or merged-blob detections, which blob size alone can flag.
On the 887 rows that match a real part (51 out of tolerance; error median 2.05 mm, p90 8.48 mm):
GB AUROC **0.966** vs quality **0.767**; risk-coverage AUC 0.0038 vs 0.0191.

# Model Card — Student Failure-Risk Classifier

| | |
|---|---|
| **Name / registry** | `student-fail-risk` (MLflow Model Registry, alias `champion`) |
| **Task** | Binary classification: will the student finish **below the pass mark** (`fail = G3 < 10`)? |
| **Intended use** | Early-warning list for tutors/advisors so they can *prioritise support*. A human decides every intervention. |
| **Not for** | Grading, admissions, discipline, or any automated decision about a student. |
| **Champion** | Logistic Regression (C=1.0, class-weighted), scenario `mid_term` (uses G1 and G2), decision threshold 0.24 |
| **Data** | UCI Student Performance (1,044 rows: 395 Mathematics + 649 Portuguese, school year 2005-06, two Portuguese schools). Fail rate 22.0 %. |
| **Reproducibility** | seed 42; dataset sha256 and git commit stored in `models/champion/metadata.json` and as MLflow tags |

## Why this target
Part 1's `risk_score` / `risk_category` is a hand-weighted formula over G1, G2, absences and study time — it never produces a "High" student (max score 63), so the high-risk mart is empty. It is also a deterministic function of the inputs, so predicting it would be circular. The real outcome the project is about is *failing the subject*, so the target is `G3 < 10`. Columns derived from G3 (`g3`, `result_status`, `performance_trend`) and Part 1's risk columns are excluded from features (leakage policy in `ml/features.py`).

## Two prediction scenarios
| Scenario | Inputs available | Use |
|---|---|---|
| `early` | G1 (+ background, absences) | Earliest warning, weaker |
| `mid_term` (**deployed**) | G1 **and** G2 | Most accurate; still leaves one period to intervene |

## Evaluation (held-out test set: 212 rows, 47 failing students)
Split: stratified by subject × outcome, **grouped by student** (382 students appear in both subject files; both rows are kept in the same split — verified `group_leakage = 0`). The UCI data has no timestamp, so a chronological split is not possible.
Threshold chosen on validation to maximise F2 (recall-weighted: missing a failing student is worse than a false alarm). Test set touched once, after model selection.

**mid_term scenario**

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | F2 |
|---|---|---|---|---|---|---|
| Prior only (baseline) | – | 0.222 | 0.500 | 0.00 | 0.00 | 0.00 |
| Rule: `G2 < 10` (baseline) | – | 0.658 | 0.907 | 0.94 | 0.69 | 0.87 |
| **Logistic Regression (champion)** | **0.914** | 0.851 | 0.944 | 0.94 | 0.58 | 0.83 |
| Random Forest | 0.911 | 0.860 | 0.963 | 0.94 | 0.68 | 0.87 |
| XGBoost | 0.909 | 0.875 | 0.963 | 0.98 | 0.58 | 0.86 |

95 % bootstrap CI for the champion on test: PR-AUC 0.75–0.92, recall 0.86–1.00, precision 0.47–0.70.

**early scenario (G1 only, comparison run — not deployed):** XGBoost PR-AUC 0.74 (rule `G1 < 10`: 0.56), recall 0.83, precision 0.57.

### What these numbers do and do not say
* **The three model families are statistically indistinguishable.** Validation PR-AUCs differ by 0.005; test differences sit well inside the confidence interval (only 47 positives). The champion rule therefore prefers the *simplest* model within 0.01 PR-AUC of the best — hence Logistic Regression, which is also the easiest to explain and the fastest to serve.
* **With G2 available, a one-line rule is hard to beat as a yes/no flag.** `G2 < 10` has the same recall as the model with *higher* precision. The model's added value is **ranking** (PR-AUC 0.85 vs 0.66), graded risk bands, and use of G1/absences/background when G2 is borderline. If only a flag were needed, the rule would do. The model matters most in the `early` scenario, where no such obvious rule exists (PR-AUC 0.74 vs 0.56).
* **Scores are not calibrated probabilities.** Class-weighted training pushes mid/high scores above the observed failure frequency (see calibration panel in `reports/curves.png`; Brier 0.10). Use `risk_band` / `predicted_fail`, or rank by score; do not read 0.80 as "80 % chance".
* Most of the signal is `g2` (permutation importance 0.54 PR-AUC vs 0.07 for `g1`, ≤0.035 for everything else).

### Slice check (test set, small samples — indicative only)
| Slice | n | failing | Recall | Precision |
|---|---|---|---|---|
| Mathematics | 81 | 27 | 1.00 | 0.68 |
| Portuguese | 131 | 20 | 0.85 | 0.47 |
| Female | 127 | 28 | 0.96 | 0.56 |
| Male | 85 | 19 | 0.89 | 0.61 |
| School GP | 161 | 33 | 0.97 | 0.58 |
| School MS | 51 | 14 | 0.86 | 0.57 |

Portuguese has lower precision (more false alarms) and recall; MS and male students have lower recall. With 14–28 positives per slice these gaps are not statistically established, but they should be re-examined as live outcomes accumulate (`monitoring` does not yet slice by group — see limitations).

## Limitations
* Small, single-year, two-school sample; behaviour on other cohorts is unknown (drift monitoring exists for exactly this reason).
* 22 % of students fail; test positives = 47, so metrics are noisy (use the CIs).
* Features include sex, family background and romantic status. They add little (importance ≤ 0.035) but are sensitive; consider removing them for production use and compare.
* G2 is required in the deployed scenario, so predictions are available only after period 2.
* "Dropout" is not observed in this dataset; G3 = 0 (53 students) is the closest proxy and is counted as failing.

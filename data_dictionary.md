# Data Dictionary — Student Performance & Dropout-Risk Prediction System

## Source: UCI Student Performance Dataset
Public dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/320/student+performance
Files: `student-mat.csv` (Math), `student-por.csv` (Portuguese). Free, public, no login required.

## fact_student_performance (grain: one row per student per subject)
| Column | Type | Description |
|---|---|---|
| student_id | varchar(20) | Synthesized standardized ID, format `<SRC>-#####` |
| subject | varchar | Subject/source the record belongs to (math / portuguese) |
| g1, g2, g3 | smallint (0-20) | First-period, second-period, final grade |
| absences | smallint | Number of school absences (raw) |
| attendance_pct | numeric | Derived: `100 * (1 - absences / max_observed_absences)` |
| avg_internal_marks | numeric | Mean of G1 and G2 |
| assignment_completion_rate | numeric | Derived proxy from `studytime` (1–4 scale) normalized to 0–100% |
| previous_semester_performance | smallint | G1, used as the prior-period baseline |
| performance_trend | varchar | improving / declining / stable, comparing G3 vs G1 |
| result_status | varchar | PASS if G3 >= 10, else FAIL |
| risk_score | numeric (0-100) | Weighted composite: 40% attendance gap + 40% marks gap + 20% assignment gap |
| risk_category | varchar | Low (0-33) / Medium (34-66) / High (67-100), derived from risk_score |

## dim_student (grain: one row per student)
| Column | Description |
|---|---|
| student_id | Standardized identifier (join key) |
| source_system | Originating dataset (math/portuguese/ERP/LMS) |
| school, sex, age, address, famsize, pstatus | Demographic attributes from source |
| medu, fedu, mjob, fjob | Parental education/occupation |
| internet | Household internet access (yes/no) |

## mart_subject_pass_fail
Subject-wise count of PASS/FAIL and pass_rate_pct.

## mart_department_semester
Aggregated average final marks, attendance, and risk score by department/subject grouping.

## mart_high_risk_students
Filtered view of fact table where risk_category = 'High', sorted by risk_score descending —
feeds the "high-risk student list" dashboard view and Part-2 model training set.

## Validation rules applied (validate_clean.py)
1. student_id standardized as `<3-letter source prefix>-<5-digit sequence>`.
2. Exact duplicate rows removed.
3. G1/G2/G3 must be within [0, 20]; out-of-range or null values rejected and logged.
4. Missing `absences` imputed with the per-source median; negative values clipped to 0.
5. Missing `studytime` imputed with per-source median.
6. Rows with all three grades missing are rejected (cannot be used for analysis).
7. All rejected rows are logged to `data/logs/rejected_records_<timestamp>.csv` with a `reject_reason`.

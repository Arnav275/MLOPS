-- Student Performance & Dropout-Risk Prediction System
-- Star schema for the analytics warehouse (PostgreSQL)

CREATE SCHEMA IF NOT EXISTS student_analytics;
SET search_path TO student_analytics;

-- =========================
-- Dimension: Student
-- =========================
CREATE TABLE IF NOT EXISTS dim_student (
    student_id              VARCHAR(20) PRIMARY KEY,
    source_system            VARCHAR(50),
    school                    VARCHAR(10),
    sex                       VARCHAR(5),
    age                       SMALLINT,
    address                   VARCHAR(5),
    famsize                   VARCHAR(10),
    pstatus                   VARCHAR(5),
    medu                      SMALLINT,
    fedu                      SMALLINT,
    mjob                      VARCHAR(20),
    fjob                      VARCHAR(20),
    internet                  VARCHAR(5),
    load_timestamp            TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Dimension: Subject
-- =========================
CREATE TABLE IF NOT EXISTS dim_subject (
    subject_id    SERIAL PRIMARY KEY,
    subject_name  VARCHAR(50) UNIQUE NOT NULL
);

-- =========================
-- Fact: Student performance (one row per student per subject/source)
-- =========================
CREATE TABLE IF NOT EXISTS fact_student_performance (
    fact_id                        SERIAL PRIMARY KEY,
    student_id                     VARCHAR(20) REFERENCES dim_student(student_id),
    subject                        VARCHAR(50),
    g1                              SMALLINT,
    g2                              SMALLINT,
    g3                              SMALLINT,
    absences                        SMALLINT,
    attendance_pct                  NUMERIC(5,2),
    avg_internal_marks              NUMERIC(5,2),
    assignment_completion_rate      NUMERIC(5,2),
    previous_semester_performance   SMALLINT,
    performance_trend               VARCHAR(20),
    result_status                   VARCHAR(10),
    risk_score                      NUMERIC(5,2),
    risk_category                   VARCHAR(10),
    load_timestamp                  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_student ON fact_student_performance(student_id);
CREATE INDEX IF NOT EXISTS idx_fact_subject ON fact_student_performance(subject);
CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_student_performance(risk_category);

-- =========================
-- Data mart: Subject-wise pass/fail
-- =========================
CREATE TABLE IF NOT EXISTS mart_subject_pass_fail (
    subject       VARCHAR(50) PRIMARY KEY,
    pass_count     INTEGER,
    fail_count     INTEGER,
    pass_rate_pct  NUMERIC(5,2),
    load_timestamp TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Data mart: Department/semester performance
-- =========================
CREATE TABLE IF NOT EXISTS mart_department_semester (
    mart_id            SERIAL PRIMARY KEY,
    department          VARCHAR(50),
    subject              VARCHAR(50),
    avg_final_marks      NUMERIC(5,2),
    avg_attendance_pct   NUMERIC(5,2),
    avg_risk_score       NUMERIC(5,2),
    student_count        INTEGER,
    load_timestamp       TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Data mart: High-risk student list
-- =========================
CREATE TABLE IF NOT EXISTS mart_high_risk_students (
    mart_id                      SERIAL PRIMARY KEY,
    student_id                    VARCHAR(20) REFERENCES dim_student(student_id),
    source_system                  VARCHAR(50),
    subject                        VARCHAR(50),
    attendance_pct                 NUMERIC(5,2),
    avg_internal_marks             NUMERIC(5,2),
    assignment_completion_rate     NUMERIC(5,2),
    risk_score                     NUMERIC(5,2),
    risk_category                  VARCHAR(10),
    result_status                  VARCHAR(10),
    load_timestamp                 TIMESTAMP DEFAULT NOW()
);

-- =========================
-- Data-quality / rejected records log (mirrors CSV log for audit in-DB)
-- =========================
CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id         SERIAL PRIMARY KEY,
    source_name     VARCHAR(50),
    reject_reason    VARCHAR(100),
    row_snapshot     JSONB,
    logged_at        TIMESTAMP DEFAULT NOW()
);

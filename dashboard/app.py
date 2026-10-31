"""
Streamlit dashboard for the Student Performance & Dropout-Risk system.
Reads from PostgreSQL if reachable, otherwise falls back to the local
analytical CSV layer so the demo still runs without a live DB connection.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.config import PG_CONN_STRING, ANALYTICAL_DATA_DIR  # noqa: E402

st.set_page_config(page_title="Student Performance & Dropout-Risk Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_data():
    tables = ["dim_student", "fact_student_performance", "mart_subject_pass_fail",
              "mart_department_semester", "mart_high_risk_students"]
    data = {}
    try:
        engine = create_engine(PG_CONN_STRING)
        with engine.connect() as conn:
            for t in tables:
                data[t] = pd.read_sql(f"SELECT * FROM student_analytics.{t}", conn)
        st.session_state["source"] = "PostgreSQL"
    except Exception:
        csv_map = {
            "dim_student": "dim_student.csv",
            "fact_student_performance": "fact_student_performance.csv",
            "mart_subject_pass_fail": "mart_subject_pass_fail.csv",
            "mart_department_semester": "mart_department_semester.csv",
            "mart_high_risk_students": "mart_high_risk_students.csv",
        }
        for t, fname in csv_map.items():
            path = ANALYTICAL_DATA_DIR / fname
            data[t] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        st.session_state["source"] = "Local CSV (analytical layer)"
    return data


data = load_data()
fact = data["fact_student_performance"]
dim_student = data["dim_student"]
subject_mart = data["mart_subject_pass_fail"]
dept_mart = data["mart_department_semester"]
high_risk = data["mart_high_risk_students"]

st.title("Student Performance & Dropout-Risk Prediction — Analytics Dashboard")
st.caption(f"Data source: {st.session_state.get('source', 'unknown')}")

if fact.empty:
    st.warning("No analytical data found. Run the pipeline (ingest -> validate_clean -> "
               "transform -> load_to_postgres) first.")
    st.stop()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
subjects = sorted(fact["subject"].dropna().unique().tolist())
selected_subjects = st.sidebar.multiselect("Subject", subjects, default=subjects)
filtered = fact[fact["subject"].isin(selected_subjects)]

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", filtered["student_id"].nunique())
col2.metric("Avg Attendance %", f"{filtered['attendance_pct'].mean():.1f}%")
col3.metric("Avg Final Marks", f"{filtered['g3'].mean() if 'g3' in filtered.columns else filtered['G3'].mean():.1f}")
col4.metric("High-Risk Students", int((filtered.get("risk_category") == "High").sum()))

st.divider()

# ---- View 1: Attendance vs Marks ----
st.subheader("1. Attendance vs Marks")
g3_col = "g3" if "g3" in filtered.columns else "G3"
fig1 = px.scatter(
    filtered, x="attendance_pct", y=g3_col, color="subject",
    hover_data=["student_id"], labels={"attendance_pct": "Attendance %", g3_col: "Final Marks"},
    trendline="ols",
)
st.plotly_chart(fig1, use_container_width=True)

# ---- View 2: Subject-wise pass/fail distribution ----
st.subheader("2. Subject-wise Pass/Fail Distribution")
if not subject_mart.empty:
    pass_col = "pass" if "pass" in subject_mart.columns else "PASS"
    fail_col = "fail" if "fail" in subject_mart.columns else "FAIL"
    melted = subject_mart.melt(
        id_vars="subject", value_vars=[pass_col, fail_col],
        var_name="result", value_name="count"
    )
    fig2 = px.bar(melted, x="subject", y="count", color="result", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

# ---- View 3: Department / semester performance ----
st.subheader("3. Department & Semester Performance")
if not dept_mart.empty:
    fig3 = px.bar(
        dept_mart, x="subject", y="avg_final_marks",
        color=dept_mart.columns[0], barmode="group",
        labels={"avg_final_marks": "Average Final Marks"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---- View 4: High-risk student list ----
st.subheader("4. High-Risk Student List")
if not high_risk.empty:
    st.dataframe(high_risk, use_container_width=True)
    st.download_button(
        "Download high-risk list (CSV)",
        high_risk.to_csv(index=False).encode("utf-8"),
        "high_risk_students.csv",
        "text/csv",
    )
else:
    st.info("No students currently flagged as high risk for the selected filters.")

# ---- View 5: Student profile drill-down ----
st.subheader("5. Student Profile & Intervention View")
student_ids = sorted(filtered["student_id"].unique().tolist())
selected_student = st.selectbox("Select a student", student_ids)
if selected_student:
    profile = filtered[filtered["student_id"] == selected_student]
    demo = dim_student[dim_student["student_id"] == selected_student]

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.write("**Demographics**")
        st.table(demo.T.rename(columns={demo.index[0] if len(demo) else 0: "value"}))
    with pcol2:
        st.write("**Performance Records**")
        st.table(profile[["subject", "attendance_pct", "avg_internal_marks",
                           "assignment_completion_rate", "risk_score", "risk_category",
                           "result_status"]])

    if (profile.get("risk_category") == "High").any():
        st.error("⚠️ This student is flagged for early academic intervention.")
    else:
        st.success("No intervention flag for this student at this time.")

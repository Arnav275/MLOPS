"""Model-driven risk view: cohort ranking + single-student what-if scoring."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard import ml_client  # noqa: E402
from ml.config import path  # noqa: E402

st.set_page_config(page_title="Failure-Risk Predictions", layout="wide")
st.title("Failure-Risk Predictions")
st.caption("Risk score that a student finishes below the pass mark (G3 < 10), from the registered champion model. The score ranks students well but is not a calibrated probability - act on the risk band.")

BAND_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#d62828"}

try:
    info, src = ml_client.model_info()
except Exception as exc:
    st.error(f"No model available. Run `python -m ml.train` first. ({exc})")
    st.stop()

m = info["metrics"]["test"]
c = st.columns(5)
c[0].metric("Model", f"{info['model_family']}")
c[1].metric("Version", info.get("model_version", "?"))
c[2].metric("Scenario", info["scenario"])
c[3].metric("Test recall", f"{m['recall']:.0%}")
c[4].metric("Test PR-AUC", f"{m['pr_auc']:.2f}")
st.caption(f"Served from: {src}. Decision threshold {info['threshold']} (tuned for recall on validation data).")

tab1, tab2 = st.tabs(["Cohort risk list", "Score a student"])

with tab1:
    mart = path("data/analytical/mart_student_risk_predictions.csv")
    if not mart.exists():
        st.info("Run `python -m ml.batch_score` to create the prediction mart.")
    else:
        df = pd.read_csv(mart)
        f1, f2, f3 = st.columns(3)
        subj = f1.multiselect("Subject", sorted(df.subject.unique()), default=sorted(df.subject.unique()))
        band = f2.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
        only_unseen = f3.checkbox("Only students NOT used in training (honest view)", value=False,
                                  help="Rows from the train/val split were seen by the model, so their scores are optimistic.")
        view = df[df.subject.isin(subj) & df.risk_band.isin(band)]
        if only_unseen and "data_split" in view:
            view = view[view.data_split == "test"]
        k = st.columns(3)
        k[0].metric("Students shown", len(view))
        k[1].metric("High risk", int((view.risk_band == "High").sum()))
        k[2].metric("Predicted to fail", int(view.predicted_fail.sum()))
        fig = px.histogram(df[df.subject.isin(subj)], x="fail_probability", color="risk_band", nbins=25,
                           color_discrete_map=BAND_COLORS, category_orders={"risk_band": ["Low", "Medium", "High"]})
        fig.update_layout(height=300, margin=dict(t=10), xaxis_title="Model risk score")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view.drop(columns=["scored_at"], errors="ignore"), use_container_width=True, height=380)
        st.download_button("Download list (CSV)", view.to_csv(index=False).encode(), "student_risk_predictions.csv", "text/csv")
        st.caption("The same file feeds Tableau: connect to `data/analytical/mart_student_risk_predictions.csv`.")

with tab2:
    st.write("Enter what is known about a student. Optional fields left blank are imputed and reported back.")
    with st.form("score"):
        a, b_, c_, d = st.columns(4)
        subject = a.selectbox("Subject", ["math", "portuguese"])
        g1 = b_.number_input("Period-1 grade (G1)", 0, 20, 10)
        g2 = c_.number_input("Period-2 grade (G2)", 0, 20, 10, disabled=info["scenario"] == "early")
        absences = d.number_input("Absences", 0, 100, 4)
        e, f, g, h = st.columns(4)
        failures = e.number_input("Past failures", 0, 4, 0)
        studytime = f.selectbox("Study time (1-4)", [1, 2, 3, 4], index=1)
        higher = g.selectbox("Wants higher education", ["yes", "no"])
        school = h.selectbox("School", ["GP", "MS"])
        go = st.form_submit_button("Predict")
    if go:
        rec = {"subject": subject, "g1": int(g1), "g2": int(g2), "absences": int(absences),
               "failures": int(failures), "studytime": int(studytime), "higher": higher, "school": school}
        try:
            res, how = ml_client.predict(rec)
            p = res["fail_probability"]
            st.markdown(f"### {res['risk_band']} risk - model score {p:.2f}")
            st.progress(min(max(p, 0.0), 1.0))
            (st.error if res["risk_band"] == "High" else st.warning if res["risk_band"] == "Medium" else st.success)(res["recommended_action"])
            st.caption(f"model v{res['model_version']} | threshold {res['threshold']} | scored via {how} | "
                       f"{len(res['imputed_fields'])} fields imputed")
        except Exception as exc:
            st.error(str(exc))

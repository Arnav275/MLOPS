"""Model monitoring: drift, data quality, live performance, service health, retraining decision."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.bundle import load_bundle  # noqa: E402
from ml.config import load_params, path  # noqa: E402
from monitoring import report as mon  # noqa: E402

st.set_page_config(page_title="Model Monitoring", layout="wide")
st.title("Model Monitoring")
params = load_params()
cfg = params["monitoring"]
STATUS_ICON = {"OK": "🟢 OK", "WARN": "🟠 WARN", "ALERT": "🔴 ALERT", "INSUFFICIENT_DATA": "⚪ not enough data"}

window = st.sidebar.selectbox("Window", ["All time", "Last 24 h", "Last 7 days", "Last 30 days"], index=0)
hours = {"All time": None, "Last 24 h": 24, "Last 7 days": 168, "Last 30 days": 720}[window]
st.sidebar.caption("Prediction log: " + str(path(cfg["db_path"])))

try:
    bundle = load_bundle(path(params["serving"]["champion_dir"]))
    preds, lab, reqs, src = mon._from_store(bundle, cfg, hours)
except Exception as exc:
    st.error(f"Cannot load model/monitoring store: {exc}")
    st.stop()

if preds.empty:
    st.info("No predictions logged yet. Start the API and send traffic, e.g.\n\n"
            "`python -m monitoring.simulate --mode normal --n 150`")
    st.stop()

rep = mon.build_report(bundle, cfg, preds, lab, reqs, src)
d = rep["retraining"]
banner = {"NO_ACTION": st.success, "INVESTIGATE": st.warning, "RETRAIN": st.error}[d["action"]]
banner(f"**Decision: {d['action']}** - model v{rep['model']['version']} ({rep['model']['family']}), "
       f"{rep['n_predictions']} predictions, age {d['model_age_days']} days.")
for t in d["triggers"]:
    st.write(f"- **{t['id']}**: {t['detail']}")
for i in d["investigate"]:
    st.write(f"- investigate: {i}")

k = st.columns(4)
k[0].metric("Feature drift", STATUS_ICON[max((x["status"] for x in rep["feature_drift"]), key=["INSUFFICIENT_DATA", "OK", "WARN", "ALERT"].index, default="INSUFFICIENT_DATA")])
k[1].metric("Prediction drift", STATUS_ICON[rep["score_drift"]["status"]])
k[2].metric("Live performance", STATUS_ICON[rep["performance"]["status"]])
k[3].metric("Service health", STATUS_ICON[rep["service"]["status"]])

t1, t2, t3, t4, t5 = st.tabs(["Feature drift", "Class & prediction distribution", "Performance", "Latency & failures", "Data quality"])

with t1:
    fd = pd.DataFrame(rep["feature_drift"])
    if fd.empty or fd.psi.isna().all():
        st.info(f"Need at least {cfg['min_samples']} predictions to measure drift.")
    else:
        fd = fd.dropna(subset=["psi"]).sort_values("psi")
        fig = px.bar(fd, x="psi", y="feature", color="status", orientation="h",
                     color_discrete_map={"OK": "#2a9d8f", "WARN": "#e9c46a", "ALERT": "#d62828"}, height=max(300, 18 * len(fd)))
        fig.add_vline(x=cfg["psi"]["warn"], line_dash="dot"); fig.add_vline(x=cfg["psi"]["alert"], line_dash="dash")
        fig.update_layout(margin=dict(t=10), xaxis_title="PSI vs. training (dotted = warn, dashed = alert)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ALERT needs PSI >= alert threshold AND a significant KS / chi-square test; PSI alone is noisy at small n.")
        feat = st.selectbox("Compare distribution", fd.sort_values("psi", ascending=False).feature.tolist())
        ref = bundle.reference["features"][feat]
        cur = preds[feat].dropna()
        if ref["type"] == "numeric":
            fig = go.Figure()
            fig.add_histogram(x=ref["values"], name="training", histnorm="probability", opacity=.6)
            fig.add_histogram(x=cur, name="live", histnorm="probability", opacity=.6)
            fig.update_layout(barmode="overlay", height=300, margin=dict(t=10))
        else:
            live = cur.astype(str).value_counts(normalize=True)
            cats = list(ref["proportions"])
            fig = go.Figure([go.Bar(x=cats, y=[ref["proportions"][c] for c in cats], name="training"),
                             go.Bar(x=cats, y=[live.get(c, 0) for c in cats], name="live")])
            fig.update_layout(barmode="group", height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

with t2:
    cd = rep["class_distribution"]
    c = st.columns(3)
    c[0].metric("Training fail rate", f"{cd['reference_fail_rate']:.1%}")
    c[1].metric("Predicted fail rate (live)", "n/a" if cd.get("predicted_fail_rate") is None else f"{cd['predicted_fail_rate']:.1%}")
    c[2].metric("Actual fail rate (labelled)", "n/a" if cd.get("actual_fail_rate") is None else f"{cd['actual_fail_rate']:.1%}",
                help=f"{cd.get('n_labels', 0)} outcomes received via /feedback; needs {cfg['retrain']['min_labels']}")
    if "band_mix" in cd:
        st.write("Risk-band mix:", cd["band_mix"])
    sd = rep["score_drift"]
    if "psi" in sd:
        st.write(f"Score distribution PSI **{sd['psi']}** ({sd['status']}); mean score {sd['cur_mean_score']} vs {sd['ref_mean_score']} at validation.")
        fig = go.Figure()
        fig.add_histogram(x=bundle.reference["score_values"], name="validation", histnorm="probability", nbinsx=20, opacity=.6)
        fig.add_histogram(x=preds.proba, name="live", histnorm="probability", nbinsx=20, opacity=.6)
        fig.update_layout(barmode="overlay", height=300, margin=dict(t=10), xaxis_title="predicted failure probability")
        st.plotly_chart(fig, use_container_width=True)

with t3:
    p = rep["performance"]
    if p["status"] == "INSUFFICIENT_DATA":
        st.info(f"{p['n_labels']} outcomes so far; need {cfg['retrain']['min_labels']} (both classes) to judge performance. "
                "Post real end-of-term results to `POST /feedback`.")
    else:
        rows = [{"metric": m_, "training test set": p["baseline"][m_], "live": p["current"][m_]} for m_ in ["recall", "precision", "pr_auc", "brier"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.write(f"Recall drop **{p['recall_drop']:+.3f}**, PR-AUC drop **{p['pr_auc_drop']:+.3f}** "
                 f"(retrain limits {cfg['retrain']['max_recall_drop']} / {cfg['retrain']['max_pr_auc_drop']}) on {p['n_labels']} outcomes.")
        st.write("Confusion:", p["confusion"])

with t4:
    s = rep["service"]
    if s["status"] == "INSUFFICIENT_DATA":
        st.info("Not enough requests yet.")
    else:
        c = st.columns(4)
        c[0].metric("Requests", s["requests"])
        c[1].metric("p95 latency", f"{s['latency_ms']['p95']} ms", help=f"warn {cfg['latency_ms']['p95_warn']} / alert {cfg['latency_ms']['p95_alert']}")
        c[2].metric("Server errors (5xx)", f"{s['error_rate_5xx']:.2%}")
        c[3].metric("Rejected inputs (4xx)", f"{s['rejected_rate_4xx']:.2%}")
        fig = px.line(reqs.assign(i=range(len(reqs))), x="i", y="latency_ms", height=250)
        fig.update_layout(margin=dict(t=10), xaxis_title="request #", yaxis_title="latency (ms)")
        st.plotly_chart(fig, use_container_width=True)

with t5:
    q = rep["data_quality"]
    st.write(f"Status: {STATUS_ICON[q['status']]}")
    if q.get("issues"):
        st.dataframe(pd.DataFrame(q["issues"]), hide_index=True)
    elif q["status"] == "OK":
        st.success("No missing-value, range or unseen-category issues versus training.")

with st.expander("Retraining criteria (params.yaml → monitoring.retrain)"):
    st.json(cfg["retrain"])
st.download_button("Download full report (JSON)", json.dumps(rep, indent=2, default=float), "monitoring_report.json")

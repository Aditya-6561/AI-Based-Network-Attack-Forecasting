"""
TRINETRA -- defender dashboard for AI-based network attack forecasting.

Runs fully offline. Accepts an IDS2018-style CSV or a raw PCAP, rebuilds the
network state timeline, rolls the world model forward K steps at every anchor,
and shows the infiltration probability timeline, the predicted MITRE ATT&CK
stage, and the traffic features driving each forecast.

    streamlit run app.py
"""

import glob
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.attack_mapping import AttackMapper
from src.explainability import IntegratedGradientsExplainer
from src.preprocessing import REQUIRED_COLUMNS, TelemetryPreprocessor
from src.scaling import LogStandardScaler
from src.state_builder import STATE_DIM, StateBuilder
from src.world_model import TemporalWorldModel

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "world_model.pt")
BENCHMARK_PATH = os.path.join(ROOT, "models", "benchmark_results.json")
DATA_DIR = os.environ.get("TRINETRA_DATA_DIR", os.path.join(ROOT, "data"))

st.set_page_config(page_title="TRINETRA - Network Attack Forecasting",
                   page_icon="R", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
def _mtime(path):
    """Cache key: retraining while the dashboard is open must invalidate the cache,
    otherwise the page keeps serving the previous run's weights and metrics."""
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource
def load_model(_mtime_key):
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = TemporalWorldModel(
        input_dim=ck.get("input_dim", STATE_DIM),
        hidden_dim=ck.get("hidden_dim", 64),
        latent_dim=ck.get("latent_dim", 32),
        forecast_steps=ck.get("forecast_steps", 6),
        num_stages=ck.get("num_stages", 7),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    scaler = LogStandardScaler.from_state_dict(ck["scaler_state"])
    return model, scaler, ck


@st.cache_data
def load_benchmarks(_mtime_key):
    if not os.path.exists(BENCHMARK_PATH):
        return None
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def ingest(path, is_pcap, max_rows, window_sec, history_steps, forecast_steps):
    """Parse a capture into sequences. Cached on the file path + parameters."""
    if is_pcap:
        from src.pcap_ingest import parse_pcap, pcap_summary
        raw = parse_pcap(path, max_packets=max_rows)
        if raw.empty:
            return None
        summary = pcap_summary(raw)
        df = TelemetryPreprocessor(fix_12h_clock=False).clean_dataframe(raw)
    else:
        summary = None
        avail = set(pd.read_csv(path, nrows=0).columns.str.strip())
        usecols = [c for c in REQUIRED_COLUMNS if c in avail] or None
        df = TelemetryPreprocessor().load_csv_file(path, nrows=max_rows, usecols=usecols)

    sb = StateBuilder(window_duration_sec=window_sec, history_steps=history_steps,
                      forecast_steps=forecast_steps, stride=1, mapper=AttackMapper())
    X, Y_state, Y_risk, Y_stage, anchors = sb.generate_sequences(df)
    if len(X) == 0:
        return None

    labels = None
    if "Label" in df.columns:
        labels = (df["Label"].astype(str).str.strip().value_counts().to_dict())
    return {"X": X, "Y_risk": Y_risk, "Y_stage": Y_stage,
            "anchors": anchors, "labels": labels, "pcap_summary": summary,
            "n_records": int(len(df))}


# --------------------------------------------------------------------------- #
st.title("TRINETRA: AI-Based Network Attack Forecasting")
st.caption("Problem Statement 26153 - temporal latent state-transition world model "
           "for predictive cyber defence. Runs fully offline, no cloud APIs.")

model, scaler, ckpt = load_model(_mtime(MODEL_PATH))
mapper = AttackMapper()
benchmarks = load_benchmarks(_mtime(BENCHMARK_PATH))

if model is None:
    st.error("No trained model at `models/world_model.pt`. "
             "Run `python training/train.py` first.")
    st.stop()

WINDOW_SEC = ckpt.get("window_sec", 10)
HISTORY_STEPS = ckpt.get("history_steps", 12)
K = ckpt.get("forecast_steps", 6)
DEFAULT_THR = float(ckpt.get("alert_threshold", 0.5))

# ---- sidebar -------------------------------------------------------------- #
st.sidebar.header("Traffic source")
uploaded = st.sidebar.file_uploader(
    "Upload flow CSV or PCAP", type=["csv", "pcap", "pcapng"],
    help="CIC-IDS2018 / CICFlowMeter CSV, or a raw packet capture.")

local_files = sorted(sum((glob.glob(os.path.join(DATA_DIR, "*." + ext))
                          for ext in ("csv", "pcap", "pcapng")), []))
selected_local = None
if local_files:
    selected_local = st.sidebar.selectbox(
        "...or pick a local capture", ["(none)"] + [os.path.basename(f) for f in local_files])
    if selected_local == "(none)":
        selected_local = None

max_rows = st.sidebar.slider("Max records to ingest", 20000, 600000, 200000, 20000)
alert_threshold = st.sidebar.slider(
    "Alert threshold", 0.05, 0.95, min(max(DEFAULT_THR, 0.05), 0.95), 0.01,
    help="Default is the F1-optimal threshold selected on the validation split.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Architecture**\n\n"
    "- State: %d-dim vector per %ds window\n"
    "- Encoder: 2-layer LSTM over W=%d windows (%ds)\n"
    "- Transition: residual latent cell, K=%d rollout (%ds)\n"
    "- Heads: state decoder, risk, ATT&CK stage\n"
    "- Explainability: Integrated Gradients"
    % (STATE_DIM, WINDOW_SEC, HISTORY_STEPS, HISTORY_STEPS * WINDOW_SEC,
       K, K * WINDOW_SEC))

# ---- resolve input -------------------------------------------------------- #
path, is_pcap, source_name = None, False, None
if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    path, is_pcap, source_name = tmp.name, suffix in (".pcap", ".pcapng"), uploaded.name
elif selected_local:
    path = os.path.join(DATA_DIR, selected_local)
    source_name = selected_local
    is_pcap = os.path.splitext(selected_local)[1].lower() in (".pcap", ".pcapng")

if path is None:
    st.info("Upload a CSV or PCAP in the sidebar to begin. "
            "Set `TRINETRA_DATA_DIR` to browse a local dataset folder instead.")
    if benchmarks:
        st.markdown("---")
        st.subheader("Benchmark results (from `models/benchmark_results.json`)")
        st.caption("Produced by `training/train.py`; not hardcoded.")
        rows = [{"Model": n,
                 "F1 (K-horizon)": round(v["k_horizon"]["f1"], 4),
                 "Precision": round(v["k_horizon"]["precision"], 4),
                 "Recall": round(v["k_horizon"]["recall"], 4),
                 "FPR": round(v["k_horizon"]["fpr"], 4),
                 "Early-warning recall": round(v["early_warning"]["early_warning_recall"], 4),
                 "Mean lead time (s)": round(v["early_warning"]["mean_lead_time_sec"], 1)}
                for n, v in benchmarks["models"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()

# ---- run the model -------------------------------------------------------- #
with st.spinner("Ingesting %s and rebuilding network state..." % source_name):
    try:
        data = ingest(path, is_pcap, max_rows, WINDOW_SEC, HISTORY_STEPS, K)
    except Exception as exc:
        st.error("Could not parse %s: %s" % (source_name, exc))
        st.stop()

if data is None:
    st.error("Not enough traffic to form a single %ds sequence "
             "(need at least %d consecutive windows). Try a larger capture."
             % (WINDOW_SEC, HISTORY_STEPS + K))
    st.stop()

X, anchors = data["X"], data["anchors"]
Xn = scaler.transform(X.reshape(-1, STATE_DIM)).reshape(X.shape)

with st.spinner("Rolling the world model forward %d steps at %d anchors..." % (K, len(X))):
    with torch.no_grad():
        _, risk_prob, stage_prob = model.predict(torch.tensor(Xn, dtype=torch.float32))
    risk = risk_prob.numpy()             # (N, K)
    stages = stage_prob.numpy()          # (N, K, S)

peak_risk = risk.max(axis=1)
has_truth = data["Y_risk"].size > 0 and data["Y_risk"].max() > 0

# ---- headline ------------------------------------------------------------- #
st.success("Ingested %s: %s records -> %d state windows -> %d forecast anchors."
           % (source_name, "{:,}".format(data["n_records"]),
              len(X) + HISTORY_STEPS + K - 1, len(X)))

if data["pcap_summary"]:
    s = data["pcap_summary"]
    st.caption("PCAP packet-level summary: %d flows over %.0fs | %d unique dst ports | "
               "TCP %d / UDP %d | mean TTL %.1f | %d retransmissions | %d fragmented"
               % (s["flows"], s["duration_sec"], s["unique_dst_ports"], s["tcp_flows"],
                  s["udp_flows"], s["mean_ttl"], s["retransmissions"], s["fragmented_pkts"]))

n_alerts = int((peak_risk >= alert_threshold).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Anchors above threshold", "%d / %d" % (n_alerts, len(X)))
c2.metric("Peak forecast risk", "%.1f%%" % (100 * peak_risk.max()))
c3.metric("Median forecast risk", "%.1f%%" % (100 * np.median(peak_risk)))

alert_idx = np.where(peak_risk >= alert_threshold)[0]
if len(alert_idx):
    first = alert_idx[0]
    dominant = int(stages[first].mean(axis=0)[1:].argmax()) + 1
    c4.metric("First alert stage", mapper.get_stage_name(dominant).split(" (")[0])
else:
    c4.metric("First alert stage", "none")

# ---- timeline ------------------------------------------------------------- #
st.markdown("---")
st.subheader("Infiltration probability across the capture")
st.caption("Each point is a K=%d-step (%ds) forward simulation from that moment. "
           "The model never sees traffic after the anchor." % (K, K * WINDOW_SEC))

fig = go.Figure()
fig.add_trace(go.Scatter(x=anchors, y=100 * peak_risk, mode="lines",
                         name="Peak forecast risk", line=dict(color="#E63946", width=2)))
fig.add_trace(go.Scatter(x=anchors, y=100 * risk[:, 0], mode="lines",
                         name="Risk at t+%ds" % WINDOW_SEC,
                         line=dict(color="#457B9D", width=1, dash="dot")))
if has_truth:
    fig.add_trace(go.Scatter(x=anchors, y=100 * data["Y_risk"].max(axis=1),
                             mode="lines", name="Ground truth (attack in horizon)",
                             line=dict(color="#2A9D8F", width=1), opacity=0.45,
                             fill="tozeroy"))
fig.add_hline(y=100 * alert_threshold, line_dash="dash", line_color="red",
              annotation_text="alert threshold")
fig.update_layout(height=380, yaxis_title="probability (%)", xaxis_title="capture time",
                  yaxis_range=[0, 103], legend=dict(orientation="h", y=1.12),
                  margin=dict(t=40, b=40))
st.plotly_chart(fig, use_container_width=True)

# ---- per-anchor inspection ------------------------------------------------ #
st.markdown("---")
st.subheader("Forecast detail and attribution")

default_idx = int(alert_idx[0]) if len(alert_idx) else int(peak_risk.argmax())
idx = st.slider("Anchor index", 0, len(X) - 1, default_idx,
                help="Position in the capture timeline to inspect.")
st.caption("Anchor time: %s" % anchors[idx])

left, right = st.columns([5, 5])

with left:
    st.markdown("**K-step rollout from this anchor**")
    horizons = [(k + 1) * WINDOW_SEC for k in range(K)]
    roll = go.Figure()
    roll.add_trace(go.Scatter(x=horizons, y=100 * risk[idx], mode="lines+markers",
                              line=dict(color="#E63946", width=3), name="forecast risk"))
    if has_truth:
        roll.add_trace(go.Scatter(x=horizons, y=100 * data["Y_risk"][idx], mode="lines",
                                  line=dict(color="#2A9D8F", dash="dash"), name="ground truth"))
    roll.add_hline(y=100 * alert_threshold, line_dash="dash", line_color="red")
    roll.update_layout(height=300, yaxis_range=[0, 103],
                       xaxis_title="seconds ahead", yaxis_title="probability (%)",
                       legend=dict(orientation="h", y=1.15), margin=dict(t=30, b=40))
    st.plotly_chart(roll, use_container_width=True)

    st.markdown("**Predicted ATT&CK stage per horizon**")
    st.caption("The risk head decides *whether* the window is hostile; the stage head "
               "decides *which* tactic it would be. The tactic column therefore reports "
               "the most likely non-benign stage, renormalised over attack stages only — "
               "otherwise it would read 'Benign' next to a high risk score. See the "
               "README limitation: this head is weak on tactics absent from training.")
    stage_rows = []
    for k in range(K):
        p = stages[idx, k]
        attack_p = p[1:]
        top_attack = int(attack_p.argmax()) + 1
        denom = float(attack_p.sum())
        stage_rows.append({
            "Horizon": "+%ds" % ((k + 1) * WINDOW_SEC),
            "Risk": "%.1f%%" % (100 * risk[idx, k]),
            "Hostile?": "yes" if risk[idx, k] >= alert_threshold else "no",
            "Most likely tactic": mapper.get_stage_name(top_attack),
            "Tactic confidence": "%.1f%%" % (100 * attack_p[top_attack - 1] / denom)
                                 if denom > 1e-9 else "n/a",
        })
    st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)

with right:
    st.markdown("**Integrated Gradients: what is driving this forecast**")
    x_one = torch.tensor(Xn[idx: idx + 1], dtype=torch.float32)
    explainer = IntegratedGradientsExplainer(model, steps=32)
    attrs = explainer.attribute(x_one)[:10]

    names = [a[0] for a in attrs][::-1]
    vals = [a[1] for a in attrs][::-1]
    bar = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=["#E63946" if v > 0 else "#457B9D" for v in vals]))
    bar.update_layout(height=360, xaxis_title="attribution to risk logit "
                                              "(red = pushes risk up)",
                      margin=dict(t=20, b=40, l=10))
    st.plotly_chart(bar, use_container_width=True)

    temporal = explainer.temporal_attribution(x_one)
    tfig = go.Figure(go.Bar(x=[-(HISTORY_STEPS - i) * WINDOW_SEC for i in range(HISTORY_STEPS)],
                            y=temporal, marker_color="#6C757D"))
    tfig.update_layout(height=200, xaxis_title="seconds before anchor",
                       yaxis_title="attribution", margin=dict(t=20, b=40))
    st.markdown("**When in the history the risk accrued**")
    st.plotly_chart(tfig, use_container_width=True)

# ---- flagged windows ------------------------------------------------------ #
st.markdown("---")
st.subheader("Flagged forecast windows")
if len(alert_idx) == 0:
    st.info("No anchor exceeded the %.2f threshold in this capture." % alert_threshold)
else:
    flagged = []
    for i in alert_idx[:200]:
        k = int(risk[i].argmax())
        # Every row here is already above the alert threshold, so the benign class
        # is excluded: the question is which tactic, not whether.
        stage_i = int(stages[i, k][1:].argmax()) + 1
        row = {
            "Anchor time": str(anchors[i]),
            "Peak risk": "%.1f%%" % (100 * risk[i, k]),
            "Peak at": "+%ds" % ((k + 1) * WINDOW_SEC),
            "Most likely tactic": mapper.get_stage_name(stage_i),
        }
        if has_truth:
            row["Ground truth"] = ("attack" if data["Y_risk"][i].max() > 0.5 else "benign")
        flagged.append(row)
    st.caption("%d anchors flagged; showing first %d." % (len(alert_idx), len(flagged)))
    st.dataframe(pd.DataFrame(flagged), use_container_width=True, hide_index=True)

# ---- ground-truth agreement ---------------------------------------------- #
if has_truth:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    yt = (data["Y_risk"] > 0.5).astype(int).ravel()
    yp = risk.ravel()
    yb = (yp >= alert_threshold).astype(int)
    st.markdown("---")
    st.subheader("Agreement with labels in this file")
    st.caption("Computed live on the loaded capture at the current threshold. "
               "For the formal held-out benchmark see the table below.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1", "%.4f" % f1_score(yt, yb, zero_division=0))
    m2.metric("Precision", "%.4f" % precision_score(yt, yb, zero_division=0))
    m3.metric("Recall", "%.4f" % recall_score(yt, yb, zero_division=0))
    m4.metric("AUROC", "%.4f" % roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else "n/a")

# ---- benchmarks ----------------------------------------------------------- #
st.markdown("---")
st.subheader("Held-out benchmark: world model vs baselines")
if not benchmarks:
    st.warning("No `models/benchmark_results.json`. Run `python training/train.py`.")
else:
    proto = benchmarks["protocol"]
    st.caption("Chronological split; test = %s. %s test sequences, positive rate %.3f. "
               "Thresholds selected on the validation split. Generated by training/train.py."
               % (", ".join(f.split("_")[0] for f in proto["test_files"]),
                  "{:,}".format(proto["n_test"]), proto["test_positive_rate"]))

    budget = proto.get("equal_fpr_budget")
    tab0, tab1, tab2 = st.tabs(
        ["Equal false-alarm budget", "K-horizon forecast (%ds)" % (K * WINDOW_SEC),
         "Early warning"])

    with tab0:
        if budget is None or "k_horizon_equal_fpr" not in next(iter(benchmarks["models"].values())):
            st.info("Rerun `training/train.py` to generate the equal-budget comparison.")
        else:
            st.caption(
                "Every model held to the same %.0f%% false-alarm budget, fixed on the "
                "validation split. Thresholds tuned for F1 transfer badly between attack "
                "families, so an equal budget compares discrimination rather than "
                "calibration drift. The trivial row is the floor: at a %.0f%% positive "
                "rate, always alerting scores F1 %.2f on its own."
                % (100 * budget, 100 * proto["test_positive_rate"],
                   2 * proto["test_positive_rate"] / (1 + proto["test_positive_rate"])))
            st.dataframe(pd.DataFrame([{
                "Model": n,
                "F1": round(v["k_horizon_equal_fpr"]["f1"], 4),
                "Precision": round(v["k_horizon_equal_fpr"]["precision"], 4),
                "Recall": round(v["k_horizon_equal_fpr"]["recall"], 4),
                "FPR": round(v["k_horizon_equal_fpr"]["fpr"], 4),
                "AUPRC": round(v["k_horizon"]["auprc"], 4),
                "AUROC": round(v["k_horizon"]["auroc"], 4),
            } for n, v in benchmarks["models"].items()]),
                use_container_width=True, hide_index=True)

    with tab1:
        st.caption("Each model at its own F1-optimal validation threshold.")
        st.dataframe(pd.DataFrame([{
            "Model": n,
            "F1": round(v["k_horizon"]["f1"], 4),
            "Precision": round(v["k_horizon"]["precision"], 4),
            "Recall": round(v["k_horizon"]["recall"], 4),
            "FPR": round(v["k_horizon"]["fpr"], 4),
            "AUPRC": round(v["k_horizon"]["auprc"], 4),
        } for n, v in benchmarks["models"].items()]),
            use_container_width=True, hide_index=True)
    with tab2:
        ew = ("early_warning_equal_fpr"
              if "early_warning_equal_fpr" in next(iter(benchmarks["models"].values()))
              else "early_warning")
        st.caption("Restricted to onset sequences: no attack in the observed history, "
                   "attack inside the forecast horizon. Persistence of an already-visible "
                   "attack cannot score here. Shown at the equal false-alarm budget. "
                   "Only %d onset sequences exist in the test split, so these figures "
                   "carry wide error bars."
                   % next(iter(benchmarks["models"].values()))["early_warning"]
                         ["n_onset_sequences"])
        st.dataframe(pd.DataFrame([{
            "Model": n,
            "Early-warning recall": round(v[ew]["early_warning_recall"], 4),
            "Mean lead time (s)": round(v[ew]["mean_lead_time_sec"], 1),
            "False alarm rate": round(v[ew]["onset_false_alarm_rate"], 4),
        } for n, v in benchmarks["models"].items()]),
            use_container_width=True, hide_index=True)

    stage_stats = benchmarks.get("attack_stage")
    if stage_stats:
        st.markdown("**ATT&CK stage head**")
        st.caption(
            "Validation accuracy on attack windows: **%.4f** (tactics seen in training). "
            "Test accuracy: **%.4f** — the test families (Infiltration, Bot) map to stages "
            "%s, none of which appear in the training stages %s, so the head cannot name "
            "them and scores 0 by construction. This is the weakest component of the "
            "system and is reported as such."
            % (stage_stats.get("val_accuracy_on_attack_windows", float("nan")),
               stage_stats.get("test_accuracy_on_attack_windows", float("nan")),
               stage_stats.get("stages_present_in_test", []),
               stage_stats.get("stages_present_in_train", [])))

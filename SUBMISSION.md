# Submission pack — PS 26153

Everything below is derived from `models/benchmark_results.json`. Regenerate it
with `python training/train.py` before recording, and read the numbers off the
file — do not retype them from here if you have rerun training.

---

## A. Five-slide technical presentation

### Slide 1 — The gap

- A flow classifier learns `P(malicious | flow)`. One flow, one label.
- That throws away the thing that defines an intrusion: **order and timing**.
  Which ports were probed first, whether SYN preceded ACK, how reconnaissance
  paced itself before lateral movement.
- An infiltration is a process unfolding over time, not an anomalous packet.
- **Ask:** predict the attacker's *next* state, not label the current one.

### Slide 2 — World model, not classifier

Show the pipeline diagram from `ARCHITECTURE.md` §2.

- Network state `S_t`: 42-D vector per 10 s window — flow-level (volumes, TCP flag
  distribution, IAT statistics) **and** packet-level (TCP window size, segment
  size floor, payload distribution, destination-port entropy and cardinality).
- Encoder: 2-layer LSTM over 12 windows (120 s) → latent `z_t ∈ ℝ³²`.
- **Transition cell `T(·)`: `z_{t+k-1} → z_{t+k}`, applied 6 times.**
- Three heads off each rolled latent: predicted state, risk, ATT&CK stage.
- **The line that matters:** during rollout the model never sees future traffic.
  At step *k* it conditions only on its own prediction. That is the difference
  between simulating a future and classifying a present.

### Slide 3 — Honest evaluation

- Split grouped by capture day; every test day strictly later than train and val.
- Test = Infiltration + Bot: **families that appear in neither other split**.
- Scaler fit on train only. Threshold and early stopping chosen on validation.
  Test scored once.
- Three tasks, hardest last:
  1. `t+1` detection — the classic IDS task.
  2. `K`-horizon forecast — attack anywhere in the next 60 s.
  3. **Early warning** — onset sequences only: benign history, attack inside the
     horizon. Persistence of a visible attack cannot score.
- **Include the trivial "always alert" row.** At a 38% positive rate it scores
  F1 ≈ 0.55 unaided. Any model near that number is not working. Saying this out
  loud is stronger than a headline F1 a judge can puncture.

### Slide 4 — Results and explainability

- Table straight from `benchmark_results.json`: trivial reference, logistic
  regression, vanilla LSTM, world model — across all three tasks.
- Lead the discussion with **early warning and FPR**, not with `t+1` F1.
- Integrated Gradients screenshot: which features drive the forecast, and when in
  the 120 s history the risk accrued.
- Attribution is taken on the risk **logit**: on the saturated sigmoid the
  gradient vanishes and attributions collapse exactly when the model is most
  confident.

### Slide 5 — Deployment and limits

- Fully offline. No cloud API anywhere. CSV **or** raw PCAP via scapy.
- Inference is one 12-step LSTM pass plus six small MLPs per anchor — a capture
  day scores in seconds on CPU. That matters for Critical Information
  Infrastructure, where shipping telemetry off-site is not permitted.
- Limits stated plainly (see `README.md` §8): window-level labels; `t+1` and
  `K`-horizon are persistence-dominated; lead time capped at the 60 s horizon;
  the stage head cannot name a tactic it never saw in training, so test-split
  stage accuracy is 0 by construction.

---

## B. Two-minute demo video script

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | `README.md` title, then `ARCHITECTURE.md` §2 diagram | "A normal IDS labels one flow at a time. TRINETRA learns how network state *transitions*, then rolls it forward to forecast an attack before it lands." |
| 0:15–0:35 | Terminal: `python training/train.py` scrolling; land on the results table | "Trained on brute force, DoS and DDoS days. Tested on Infiltration and Bot — attack families it has never seen. Every number here is written to `benchmark_results.json`; nothing is hardcoded." |
| 0:35–0:50 | Point at the trivial-reference row | "We report a trivial always-alert predictor as the floor. On this split it scores F1 around 0.55 by itself. This is what any real result has to clear." |
| 0:50–1:05 | `streamlit run app.py`; pick the CSV from the sidebar | "The dashboard runs fully offline. Load an IDS2018 CSV — or a raw PCAP, parsed with scapy." |
| 1:05–1:25 | Risk timeline; hover a spike | "Each point is a 60-second forward simulation from that moment. The model never sees traffic after the anchor. Green is ground truth for comparison." |
| 1:25–1:45 | Rollout chart + ATT&CK stage table | "Per horizon: infiltration probability and the predicted ATT&CK tactic — from the forecast latent, not a lookup of the current label." |
| 1:45–2:00 | Integrated Gradients panels | "And for every forecast: which traffic features drove it, and when in the two-minute history the risk built up. No black-box scores." |

**Recording notes**

- Pre-run training so the results table is on screen instantly; do not record the
  10–20 minute wait.
- Pre-load the CSV once before recording so Streamlit's cache is warm.
- **Do not demo the drag-and-drop uploader with a full IDS2018 CSV.** Start the
  app with TRINETRA_DATA_DIR pointed at your dataset folder and pick the file from
  the sidebar dropdown -- it reads from disk with no size limit and loads in
  seconds. Uploading a 380 MB file buffers it in memory and is slow on camera.
- `data/demo_capture.pcap` is a small synthetic capture (benign web traffic, a
  horizontal port scan, retransmissions and fragments) for demonstrating the PCAP
  path quickly. It appears in the sidebar dropdown alongside the CSVs.
- Kill any old `streamlit` process before recording. Streamlit caches the model and
  metrics per process, so a server left running from before a retrain will show the
  previous run's numbers.

---

## C. Deliverables checklist

| Required | Where | Status |
|---|---|---|
| Source code | this repository | done |
| README with setup | `README.md` | done |
| Architecture document (≤2 pages) | `ARCHITECTURE.md` | done |
| Feature extraction pipeline (CSV + PCAP) | `src/preprocessing.py`, `src/state_builder.py`, `src/pcap_ingest.py` | done |
| Trained world model + weights + reproducible config | `models/world_model.pt`, `configs/config.yaml`, `training/train.py` | done |
| K-step forward simulation engine | `TemporalWorldModel.forward` / `.predict` | done |
| ATT&CK stage mapping | `configs/attack_mapping.yaml`, stage head | done |
| Explainability | `src/explainability.py` (Integrated Gradients) | done |
| Offline demo interface | `app.py` (Streamlit, CSV + PCAP upload) | done |
| Benchmark vs logistic regression | `models/benchmark_results.json` | done |
| Demo video (≤2 min) | script in §B | **to record** |
| Technical presentation (≤5 slides) | content in §A | **to build** |

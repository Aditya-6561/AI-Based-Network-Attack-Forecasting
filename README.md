# TRINETRA — AI-Based Network Attack Forecasting from Network Traffic Data

**Problem Statement 26153.** A temporal *world model* that learns the transition
dynamics of network state, `P(S_{t+1} | S_t)`, and rolls them forward to forecast
attacker progression **before** the kill chain completes — rather than classifying
each flow in isolation.

Fully open-source. Runs entirely offline; no cloud APIs.

---

## 1. What this is (and is not)

A conventional IDS maps one flow to one benign/malicious label, discarding the
temporal structure of an intrusion. TRINETRA instead:

1. Aggregates raw telemetry into a **42-dimensional network state vector** `S_t`
   per 10-second window (flow-level *and* packet-level attributes).
2. Encodes an observed history `S_{t-11..t}` (120 s) into a latent state `z_t`.
3. **Advances the latent state autoregressively** through a learned transition
   cell `T(·)` for `K = 6` steps (60 s). Ground-truth future states are never fed
   back — at step *k* the model conditions only on its own prediction `z_{t+k-1}`.
4. Decodes each rolled-out latent into a predicted observable state, an
   infiltration probability, and a **MITRE ATT&CK stage**.
5. Attributes every forecast back onto specific traffic features via **Integrated
   Gradients**.

The transition cell is what makes this a world model: at inference the system
simulates a future that has not been observed.

---

## 2. Setup

```bash
pip install -r requirements.txt
```

Download the CSE-CIC-IDS2018 `TrafficForML_CICFlowMeter` CSVs (9 files) and point
the pipeline at them:

```bash
export TRINETRA_DATA_DIR=/path/to/ids2018      # Windows: set TRINETRA_DATA_DIR=...
```

### Train and benchmark

```bash
python training/train.py --max-rows 600000 --stride 1 --epochs 30
```

Writes `models/world_model.pt` and `models/benchmark_results.json`. Deterministic
(`seed = 1337`). Runs on CPU in roughly 10–20 minutes; CUDA is used automatically
when available.

### Launch the dashboard

```bash
streamlit run app.py
```

### Loading large captures — point at the folder, don't upload

The IDS2018 CSVs are 200–380 MB each. **Set `TRINETRA_DATA_DIR` instead of using the
uploader.** This reads straight from disk, has no size limit, and is the intended
path:

```bash
export TRINETRA_DATA_DIR=/path/to/ids2018
streamlit run app.py
```

On Windows (cmd):

```bash
set TRINETRA_DATA_DIR=C:\Users\you\trinetra-data\ids2018 && streamlit run app.py
```

Every `.csv`, `.pcap` and `.pcapng` in that folder then appears in the sidebar
dropdown. Nothing is copied and nothing is uploaded.

Memory is bounded by the **"Max records to ingest"** slider, not by file size:
reading 600,000 rows from a 383 MB CSV takes about 9 s and peaks near 1 GB of RAM,
because only the 38 columns the state builder needs are parsed and the reader stops
at the row cap. A 383 MB file and a 105 MB file cost the same at the same row count.

Drag-and-drop upload still works for a one-off file a reviewer brings along, and
`.streamlit/config.toml` raises Streamlit's 200 MB default cap to 1 GB. Uploads are
buffered in memory, so the folder route is better for anything large.

`data/demo_capture.pcap` is a small synthetic capture (benign web traffic, a
horizontal port scan, retransmissions and IP fragments) included so the PCAP path
can be demonstrated without a large capture file.

---

## 3. Evaluation protocol

The split is **grouped by capture day**. Every test day is strictly later than
every train and validation day, and the test attack families appear in neither:

| Split | Days | Attack families |
|---|---|---|
| Train | 14, 15, 21, 22 Feb | FTP brute force, DoS (GoldenEye, Slowloris), DDoS (HOIC, LOIC), web brute force / XSS / SQLi |
| Validation | 16, 23 Feb | DoS (Hulk, SlowHTTPTest), web attacks |
| **Test** | 28 Feb, 1, 2 Mar | **Infiltration, Bot — unseen in training** |

Validation deliberately holds out a day with volumetric attacks. Built only from
the web-attack days, validation carries a few hundred hostile flows in 600k and
has almost no signal at 10 s granularity — early stopping then degenerates into
picking an undertrained epoch, which is exactly what happened in an earlier run.

Rules enforced in `training/train.py`:

- The scaler is fit on **training data only**.
- Alert thresholds are chosen on the **validation split**, never on test.
- Model selection (early stopping) uses **validation AUPRC**, not test.
- The test split is scored once, and every number in the dashboard is read from
  `models/benchmark_results.json`. **No metric is hardcoded anywhere.**

### Three tasks are reported

1. **`t+1` detection** — is the next window hostile? (what a classic IDS does)
2. **`K`-horizon forecast** — is any of the next 60 s hostile?
3. **Early warning** — the honest forecasting test. Restricted to *onset*
   sequences: **no attack anywhere in the observed history**, but an attack
   appears inside the forecast horizon. An already-visible attack cannot score
   here, so persistence is excluded and only genuine prediction counts.
   Lead time is `(index of first future attack window + 1) × 10 s`, bounded above
   by the 60 s horizon.

A **trivial "always alert" predictor** is reported alongside the models. On a test
split with a 38% positive rate it already scores F1 ≈ 0.55, so any model not
clearly above it is not doing useful work. Reporting it makes that impossible to
hide behind a headline F1.

### Results (test split: 28 Feb, 1 Mar, 2 Mar — Infiltration + Bot)

All models held to the **same 5% validation false-alarm budget**, so the comparison
measures discrimination rather than how well each model's threshold happens to
transfer across attack families. 9918 test sequences, positive rate 0.3804.

| Model | F1 | Precision | Recall | FPR | AUPRC | AUROC | EW recall | EW lead (s) |
|---|---|---|---|---|---|---|---|---|
| Always-alert (trivial reference) | 0.5512 | 0.3804 | 1.0000 | 1.0000 | 0.3804 | 0.5000 | 1.00 | 35.0 |
| Logistic Regression (static S_t) | 0.3116 | 0.6807 | 0.2020 | 0.0582 | 0.6175 | 0.7227 | 0.10 | 50.0 |
| Vanilla LSTM classifier | 0.4874 | 0.6251 | 0.3995 | 0.1471 | 0.6465 | 0.7670 | 0.47 | 37.9 |
| Temporal World Model (K-step rollout) | 0.4954 | 0.8117 | 0.3565 | 0.0508 | 0.6931 | 0.7694 | 0.10 | 50.0 |

Read this table as follows.

- **The trivial row is the floor.** "Always alert" scores F1 0.55 on its own
  because 38% of test windows are hostile. A high F1 alone proves nothing here.
- **AUPRC and AUROC are the threshold-free measures.** The world model leads both
  (AUPRC 0.693 vs 0.647 for the LSTM and 0.618 for logistic regression), i.e. it
  ranks hostile windows better than either baseline.
- **At an equal false-alarm budget the world model has by far the best
  precision** (0.81 vs 0.63) and is the only learned model that actually stays
  inside the 5% budget on test — the LSTM overshoots to 14.7%. For a SOC, that
  difference is the difference between a usable alert queue and an ignored one.
- **The LSTM still wins early-warning recall** (0.47 vs 0.10), at roughly triple
  the false-alarm rate. This is an honest weakness, not a rounding artifact: the
  world model is calibrated conservatively and misses onsets the LSTM catches.
  Note the early-warning sample is only 30 onset sequences, so both figures carry
  wide error bars.

Numbers regenerate with `python training/train.py`; they are read from
`models/benchmark_results.json` and are not hardcoded anywhere.

---

## 4. Network state vector (42 dimensions)

All 42 dimensions are computed from data; none are placeholders.

**Flow-level (NetFlow/IPFIX-style)** — flow count; forward/backward packet and
byte totals; flow duration mean and standard deviation; bytes/s and packets/s;
per-window SYN, ACK, RST, FIN, PSH, URG counts; SYN:ACK ratio; flow IAT mean,
std and max; forward and backward IAT means; forward and backward packets/s;
subflow packet and byte counts; down/up ratio; protocol mix (TCP and UDP shares).

**Packet-level (PCAP-derived / CICFlowMeter packet statistics)** — TCP initial
window size (forward and backward); minimum forward segment size; payload size
distribution (mean, std, variance, average packet size); active and idle times;
**destination-port Shannon entropy** and **unique destination-port count** — the
horizontal port-scan signature; a silent-window flag.

When ingesting a PCAP directly, `src/pcap_ingest.py` additionally measures TTL
mean and variance, IP fragmentation counts, and TCP retransmissions, which the
CSVs do not expose.

---

## 5. Architecture

```
flows ──► StateBuilder ──► S_{t-11..t} ──► LSTM Encoder ──► z_t
                                                             │
                              ┌──────────────────────────────┘
                              ▼
                        Transition T(·)  ──►  z_{t+1} ──► T(·) ──► z_{t+2} ··· z_{t+6}
                              │                  │                      │
                              └──────────────────┴──────────────────────┘
                                                 ▼
                              ┌──────────────────┼──────────────────┐
                              ▼                  ▼                  ▼
                       State decoder       Risk head          ATT&CK stage head
                        Ŝ_{t+k}           P(attack)          7-way tactic
```

- **Encoder** — 2-layer LSTM (hidden 64) → `tanh` latent, dim 32.
- **Transition** — residual MLP `z + f(z)` with LayerNorm; the residual form keeps
  the 6-step rollout numerically stable.
- **Heads** — state decoder (MSE), risk (BCE-with-logits, class-weighted), ATT&CK
  stage (cross-entropy over 7 tactics).
- **Loss** — `L = α·L_state + β·L_risk + γ·L_stage` with `α=1, β=2, γ=1`.

~73k parameters. Small by design: the test is whether *learned dynamics* beat a
classifier, not whether a bigger network wins.

### Why signed-log scaling matters

Flow telemetry is heavy-tailed — `Flow Duration` and byte counters span ~8 orders
of magnitude. A plain `StandardScaler` leaves 99% of windows squashed near zero
with a few 100-σ outliers, which saturates the LSTM and collapses the risk head to
the majority class. `src/scaling.py` applies `sign(x)·log1p(|x|)` before
standardising.

---

## 6. Repository layout

```
trinetra-world-model/
├── app.py                      # Streamlit defender dashboard (CSV + PCAP upload)
├── requirements.txt
├── ARCHITECTURE.md             # 2-page architecture document
├── configs/
│   ├── config.yaml             # windowing, model and loss hyperparameters
│   └── attack_mapping.yaml     # dataset label -> MITRE ATT&CK tactic + stage index
├── src/
│   ├── preprocessing.py        # ingest, 12-hour clock repair, NaN/Inf handling
│   ├── state_builder.py        # 42-D windowed state vectors + sequence tensors
│   ├── scaling.py              # signed-log1p standardisation
│   ├── world_model.py          # encoder / transition / decoder / risk / stage heads
│   ├── attack_mapping.py       # ATT&CK mapping engine
│   ├── explainability.py       # Integrated Gradients attribution
│   └── pcap_ingest.py          # scapy PCAP -> flow records (+ TTL, retrans, frags)
├── training/
│   └── train.py                # training + benchmark, writes benchmark_results.json
└── models/
    ├── world_model.pt          # weights + scaler state + selected threshold
    ├── benchmark_results.json  # every reported metric
    └── train_log.txt           # full training console log
```

---

## 7. Two data-quality issues worth knowing about

Both are handled in `src/preprocessing.py`, and both silently corrupt results if
left unfixed:

1. **CIC-IDS2018 timestamps use a 12-hour clock with no AM/PM marker.** Parsed
   naively, `01:48:53` (1:48 PM) becomes 1:48 AM, and sorting places the afternoon
   *before* the morning — so a model trained on it learns dynamics across a
   timeline that jumps 5 PM → 8 AM mid-sequence. Hours below 8 are shifted by +12
   (captures run ~08:00–17:30), guarded so the shift only applies when the file
   genuinely looks 12-hour.
2. **A handful of rows carry epoch-zero (1970) timestamps.** At 10 s resolution a
   single one stretches the resample span across ~48 years, producing ~10⁸ empty
   buckets and exhausting memory. Rows more than 2 days from the median timestamp
   are dropped; `StateBuilder` also raises a clear error above `MAX_WINDOWS`.

---

## 8. Limitations

- Windows are labelled hostile if **any** flow in them is hostile, so a window
  mixing benign and attack traffic is scored entirely as attack.
- Attack episodes in IDS2018 run for hours, so the `t+1` and `K`-horizon tasks are
  dominated by persistence. The **early-warning** metric exists precisely because
  those two numbers overstate forecasting ability; read it first.
- Lead time is capped at the 60 s horizon by construction. It is not a claim about
  arbitrary-horizon prediction.
- Source IP/destination IP are dropped so the model cannot memorise host
  signatures; this also means it cannot use host reputation.
- Single dataset. Generalisation is tested across *unseen attack families* within
  IDS2018, not across networks.
- **The ATT&CK stage head is the weakest component and should be presented as
  such.** It can only emit tactics it saw in training, and Infiltration and Bot
  are absent from the training days, so test-split stage accuracy is 0 by
  construction. Even on validation, where the tactics *were* seen, accuracy is
  only 0.053 — inverse-frequency class weighting lifted it off zero but did not
  make it good. Treat the stage output as an experimental signal, not a claim.
- Only the first 600k rows of each CSV are read. That spans a full capture day but
  not every flow within it.

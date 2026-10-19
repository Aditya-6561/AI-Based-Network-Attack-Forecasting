# TRINETRA — Architecture Document

**AI-Based Network Attack Forecasting from Network Traffic Data (PS 26153)**

---

## 1. Problem framing

A conventional IDS learns `P(malicious | flow)` — one flow, one label. That
discards the structure that defines an intrusion: the order in which ports are
probed, the way SYN floods precede ACK floods, the inter-arrival timing of
reconnaissance before lateral movement. An infiltration is a *process*.

TRINETRA learns **transition dynamics** instead:

```
P(S_{t+1} | S_t)          where S_t is the observed state of the network at time t
```

Once those dynamics are learned, the system can be run forward from a current
observation to simulate futures that have not happened yet, and can raise an
alarm about a state the attacker has not reached.

---

## 2. Pipeline

```
 raw telemetry            state construction              world model                 decision support
┌──────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐
│ IDS2018 CSV  │   │ 10 s resample            │   │ LSTM encoder        │   │ risk timeline           │
│      or      │──►│ 42-D state vector S_t    │──►│  → z_t (dim 32)     │──►│ ATT&CK stage per step   │
│ PCAP (scapy) │   │ flow + packet features   │   │ transition T(·) ×6  │   │ Integrated Gradients    │
└──────────────┘   │ ATT&CK stage per window  │   │ 3 prediction heads  │   │ flagged windows         │
                   └──────────────────────────┘   └─────────────────────┘   └─────────────────────────┘
```

### 2.1 State representation

Traffic is resampled into fixed 10 s windows. Each window becomes a
42-dimensional vector combining two levels of observation, because they expose
different attacker behaviour:

- **Flow-level** captures aggregate behaviour — a SYN flood shows up as volume
  and flag-count anomalies: packet/byte totals, flow duration statistics, per-window
  SYN/ACK/RST/FIN/PSH/URG counts, SYN:ACK ratio, IAT mean/std/max, subflow
  counters, down/up ratio, TCP and UDP protocol shares.
- **Packet-level** exposes timing and sequencing that flow thresholds miss — a
  slow reconnaissance scan: TCP initial window size (both directions), minimum
  forward segment size, payload size distribution, active/idle times, and
  **destination-port Shannon entropy plus unique-port cardinality**, which is the
  horizontal port-scan signature.

Ingesting a PCAP directly adds TTL mean/variance, IP fragmentation counts and TCP
retransmission counts, which the CICFlowMeter CSVs do not carry.

Source and destination IPs are dropped so the model cannot memorise host
identities and pass that off as generalisation.

A sequence example is `W = 12` observed windows (120 s of context) paired with the
`K = 6` windows that follow (60 s of forecast target).

### 2.2 World model

| Component | Form | Role |
|---|---|---|
| Encoder | 2-layer LSTM (hidden 64) → `tanh` linear | history `S_{t-11..t}` → latent `z_t ∈ ℝ³²` |
| Transition | residual MLP `z + f(z)`, LayerNorm | **`z_{t+k-1} → z_{t+k}`** |
| State decoder | 2-layer MLP | `z_{t+k} → Ŝ_{t+k} ∈ ℝ⁴²` |
| Risk head | 2-layer MLP → logit | `z_{t+k} → P(attack at t+k)` |
| Stage head | 2-layer MLP → 7 logits | `z_{t+k} → ATT&CK tactic at t+k` |

~73k parameters total. Deliberately small: the claim under test is that *learned
dynamics* beat a classifier, not that a larger network wins.

**The rollout is strictly autoregressive.** At every step `k` the transition cell
consumes only its own previous output; no ground-truth future state is ever
injected. This is what separates a world model from a sequence classifier with a
6-way output, and it is why the same weights can be rolled further than they were
trained.

The residual form `z + f(z)` matters: an unconstrained MLP applied six times
either saturates or diverges, and the risk curve degenerates into a step function.

**Loss** — `L = α·L_state(MSE) + β·L_risk(BCE, class-weighted) + γ·L_stage(CE)`
with `α = 1, β = 2, γ = 1`. The state-reconstruction term is what forces `T(·)` to
model dynamics rather than collapsing to a risk shortcut.

### 2.3 Scaling

Flow telemetry is heavy-tailed: `Flow Duration` and byte counters span ~8 orders
of magnitude. Plain standardisation leaves nearly all windows squashed near zero
with a handful of 100-σ outliers; the LSTM saturates and the risk head collapses
to the majority class. A signed `log1p` compression precedes standardisation
(`src/scaling.py`), fit on training data only.

### 2.4 ATT&CK mapping

`configs/attack_mapping.yaml` maps each dataset label to a MITRE tactic and stage
index: Reconnaissance (TA0043), Credential Access (TA0006), Initial Access
(TA0001), Impact (TA0040), Lateral Movement (TA0008), Command and Control
(TA0011). Each window inherits the highest stage present in it, and the stage head
is trained on the **future** window's stage — so the tactic shown for `t+40 s` is
a prediction about the future, not a lookup of the present.

### 2.5 Explainability

Integrated Gradients attributes the forecast risk back onto the `12 × 42` input
history along a straight-line path from a baseline (the training-median window) to
the observed input, using a 32-step midpoint rule.

Attribution is taken against the risk **logit**, not the sigmoid output: once the
sigmoid saturates its gradient vanishes and every attribution collapses toward
zero — the failure mode that makes saliency maps look empty exactly when the model
is most confident.

Two views are produced: which *features* drive the forecast, and *when* in the
120 s history the risk accrued.

---

## 3. Evaluation

Split is **grouped by capture day**, with every test day strictly later than every
train and validation day. Test contains Infiltration and Bot, families absent from
both other splits, so the model is measured on unseen attack behaviour rather than
memorised signatures. Validation holds out a day with volumetric attacks, because
a validation split made only of low-volume web attacks carries too little signal
to select a model or a threshold.

Scaler fit on train only; thresholds and early stopping selected on validation;
test scored once. Every number in the dashboard is read from
`models/benchmark_results.json`, written by `training/train.py`.

Three tasks, in increasing order of difficulty:

1. **`t+1` detection** — the classic IDS task.
2. **`K`-horizon forecast** — attack anywhere in the next 60 s.
3. **Early warning** — restricted to *onset* sequences: no attack anywhere in the
   observed history, attack inside the horizon. Persistence of an already-visible
   attack cannot score, so only genuine prediction counts.

Tasks 1 and 2 are dominated by persistence, because IDS2018 attack episodes run
for hours. Task 3 is the honest measure of forecasting ability, and it is the one
the baselines cannot win by construction — a static classifier scored on the
current window has no mechanism to alert before onset.

A **trivial "always alert" predictor** is reported alongside every model. At a 38%
positive rate it scores F1 ≈ 0.55 on its own, so it is the floor any real result
must clear — reporting it prevents a degenerate high-recall model from looking
successful.

Baselines: logistic regression on the current state `S_t`, and a vanilla LSTM
sequence classifier over the same 120 s history. Both receive identical features
and identical normalisation; the only difference is the absence of a learned
latent transition.

---

## 4. Deployment

Fully offline — no cloud API is contacted at any stage. The dashboard
(`streamlit run app.py`) accepts a CSV or PCAP upload, rebuilds the state
timeline, rolls the model forward at every anchor, and renders the infiltration
probability timeline, per-horizon ATT&CK stage, feature attribution, and the list
of flagged windows.

Inference is a 12-step LSTM pass plus 6 small MLP applications per anchor, so a
full capture day scores in seconds on CPU — the relevant property for a sensor
sitting on enterprise or Critical Information Infrastructure traffic, where
sending telemetry off-site is not an option.

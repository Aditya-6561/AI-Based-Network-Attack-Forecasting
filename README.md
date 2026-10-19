# AI-Based Network Attack Forecasting

A research prototype for forecasting the progression of network attacks from time-ordered traffic telemetry. Instead of making a decision about each flow in isolation, the system models how network conditions evolve and predicts whether hostile activity may appear in the next several time windows.

> **Problem Statement:** SIH 2026 — PS26153  
> **Project note:** This repository adapts an existing open-source implementation. See the [upstream project repository](https://github.com/Nikethan10/trinetra-sih2026-ps26153) for the original implementation, and retain any applicable license and attribution notices.

The pipeline is designed to run locally and does not require cloud APIs.

---

## Overview

Traditional intrusion-detection systems often classify traffic one flow at a time. That can overlook the sequence of events leading up to an attack. This project explores a temporal approach:

1. Convert traffic telemetry into a **42-feature network-state vector** for each 10-second interval.
2. Encode the latest 12 windows (a 120-second history) into a compact latent representation.
3. Roll the representation forward for six prediction steps, covering the next 60 seconds. Each step uses the model's own previous prediction rather than the ground-truth future state.
4. Decode the forecast into an estimated future state, attack risk, and a MITRE ATT&CK tactic-stage prediction.
5. Use **Integrated Gradients** to help identify which traffic features contributed to a forecast.

The learned transition module is the core forecasting component: during inference, it simulates possible future network states that have not yet been observed.

---

## Requirements and setup

### 1. Install dependencies

From the project directory, run:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the nine CSV files from the **CSE-CIC-IDS2018 `TrafficForML_CICFlowMeter`** dataset. Set an environment variable to the directory containing those files.

Linux/macOS:

```bash
export TRINETRA_DATA_DIR=/path/to/ids2018
```

Windows Command Prompt:

```bat
set TRINETRA_DATA_DIR=C:\path\to\ids2018
```

### 3. Train and evaluate

```bash
python training/train.py --max-rows 600000 --stride 1 --epochs 30
```

The training script writes the model to `models/world_model.pt` and evaluation output to `models/benchmark_results.json`. The configured random seed is `1337`. Training is intended to run on CPU and uses CUDA automatically when available.

### 4. Start the dashboard

```bash
streamlit run app.py
```

### Working with large capture files

The IDS2018 CSV files can be hundreds of megabytes each. For large datasets, configure `TRINETRA_DATA_DIR` instead of uploading files through the dashboard:

```bash
export TRINETRA_DATA_DIR=/path/to/ids2018
streamlit run app.py
```

On Windows Command Prompt:

```bat
set TRINETRA_DATA_DIR=C:\Users\you\trinetra-data\ids2018 && streamlit run app.py
```

CSV, PCAP, and PCAPNG files in that directory can then be selected from the sidebar. The directory-based workflow reads the files in place rather than copying or uploading them.

Memory use is primarily controlled by the dashboard's **Max records to ingest** setting. The reader parses only the 38 columns needed by the state builder and stops at the configured row limit. The project documentation reports that reading 600,000 rows from a 383 MB CSV takes about nine seconds and peaks near 1 GB of RAM; actual performance depends on the machine and storage.

Uploads remain available for small, one-off files. The Streamlit configuration raises the default upload limit to 1 GB, but uploads are buffered in memory, so the directory workflow is preferable for large captures.

A small synthetic capture, `data/demo_capture.pcap`, is included to demonstrate the PCAP ingestion path with benign web traffic, a horizontal port scan, retransmissions, and IP fragments.

---

## Evaluation methodology

The evaluation uses capture-day groups to separate training, validation, and testing. Test days occur later than the training and validation days, and the test set includes attack families withheld from training.

| Split | Capture days | Attack families |
|---|---|---|
| Training | 14, 15, 21, 22 February | FTP brute force; DoS (GoldenEye, Slowloris); DDoS (HOIC, LOIC); web brute force, XSS, SQL injection |
| Validation | 16 and 23 February | DoS (Hulk, SlowHTTPTest) and web attacks |
| Test | 28 February, 1 and 2 March | Infiltration and Bot — not present in training |

Validation is intentionally separated from the test days. The project notes that validation built mainly from web-attack days contains relatively few hostile flows at a 10-second resolution, which can make early stopping unreliable if the split is not chosen carefully.

The training pipeline applies these safeguards:

- The feature scaler is fitted using training data only.
- Alert thresholds are selected on validation data, not test data.
- Early stopping and model selection use validation AUPRC.
- The test set is evaluated once; reported dashboard metrics are loaded from `models/benchmark_results.json` rather than hardcoded.

### Tasks measured

1. **Next-window detection (`t+1`):** determines whether the immediately following window is hostile.
2. **60-second horizon forecast (`K` steps):** predicts whether any window in the next 60 seconds will contain hostile activity.
3. **Early warning:** evaluates only sequences with no attack in the observed history but with an attack appearing during the forecast horizon. This excludes cases where an attack is already underway. Lead time is calculated as `(first future attack window index + 1) × 10 seconds`, up to the 60-second horizon.

An always-alert predictor is included as a simple reference point. Because the test set has a substantial positive rate, this baseline can achieve a seemingly respectable F1 score without useful discrimination. Comparing against it helps put the learned models' results in context.

### Reported test results

The table below reproduces the results recorded in the project's benchmark output for the test dates listed above (Infiltration and Bot). All models use the same 5% validation false-alarm budget. The reported test set contains 9,918 sequences, with a positive rate of 0.3804.

| Model | F1 | Precision | Recall | FPR | AUPRC | AUROC | Early-warning recall | Mean lead (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-alert reference | 0.5512 | 0.3804 | 1.0000 | 1.0000 | 0.3804 | 0.5000 | 1.00 | 35.0 |
| Logistic regression (static state) | 0.3116 | 0.6807 | 0.2020 | 0.0582 | 0.6175 | 0.7227 | 0.10 | 50.0 |
| LSTM classifier | 0.4874 | 0.6251 | 0.3995 | 0.1471 | 0.6465 | 0.7670 | 0.47 | 37.9 |
| Temporal world model (multi-step rollout) | 0.4954 | 0.8117 | 0.3565 | 0.0508 | 0.6931 | 0.7694 | 0.10 | 50.0 |

**How to interpret these results**

- The always-alert baseline has an F1 of 0.5512 because 38% of test sequences are positive; F1 alone is therefore not enough to demonstrate forecasting quality.
- The temporal model has the highest AUPRC (0.6931) and AUROC (0.7694) among the learned models in this reported run.
- Its precision is 0.8117, and its test false-positive rate is 0.0508. The LSTM classifier has higher early-warning recall (0.47 compared with 0.10), but also a higher false-positive rate.
- The early-warning subset contains only 30 onset sequences, so those recall and lead-time estimates are uncertain and should be treated cautiously.

To regenerate the reported evaluation, run the training command and inspect `models/benchmark_results.json`. Results can vary if the data, environment, or implementation changes.

---

## Network-state representation

Each 10-second window is represented by 42 computed features.

**Flow-level features** include flow counts; forward and backward packet and byte totals; average and standard deviation of flow duration; byte and packet rates; TCP SYN, ACK, RST, FIN, PSH, and URG counts; SYN-to-ACK ratio; flow inter-arrival-time statistics; forward/backward inter-arrival times and packet rates; subflow packet and byte totals; down/up ratio; and TCP/UDP protocol proportions.

**Packet-level features** include initial TCP window sizes in both directions, minimum forward segment size, payload-size statistics, average packet size, active and idle durations, destination-port Shannon entropy, number of unique destination ports, and a silent-window indicator.

When PCAP files are processed directly, `src/pcap_ingest.py` additionally extracts TTL mean and variance, IP-fragment counts, and TCP retransmission information. These attributes are not available in the same form in the CSV input.

---

## Model design

The processing path is:

```text
Traffic flows
    |
    v
State builder --> 12-window history --> LSTM encoder --> latent state z_t
                                                        |
                                                        v
                                      Transition module, repeated 6 times
                                                        |
                                                        v
                                  Future latent states z_(t+1) ... z_(t+6)
                                                        |
                           +----------------------------+-------------------+
                           |                            |                   |
                           v                            v                   v
                    State decoder                  Risk head         ATT&CK head
                 Predicted network state       Attack probability   7-way tactic
```

The main components are:

- **Encoder:** two-layer LSTM with hidden size 64, producing a 32-dimensional latent representation through `tanh`.
- **Transition module:** residual MLP of the form `z + f(z)`, followed by LayerNorm. The residual connection is intended to help stabilize the six-step rollout.
- **Prediction heads:** a state decoder trained with mean squared error, a risk head trained with class-weighted binary cross-entropy, and a seven-class ATT&CK tactic head trained with cross-entropy.
- **Combined loss:** `L = α·L_state + β·L_risk + γ·L_stage`, with `α = 1`, `β = 2`, and `γ = 1`.

The network has approximately 73,000 parameters. The emphasis is on testing whether learned temporal dynamics add value beyond a static classifier, rather than increasing model size.

### Feature scaling

Network-flow measurements are heavy-tailed: values such as flow duration and byte counts can span several orders of magnitude. Standardizing raw values alone can leave most observations compressed near zero while extreme values dominate.

To reduce this effect, `src/scaling.py` applies the signed-log transform `sign(x) · log1p(|x|)` before standardization.

---

## Project structure

```text
AI-Based-Network-Attack-Forecasting/
├── app.py
├── requirements.txt
├── ARCHITECTURE.md
├── configs/
│   ├── config.yaml
│   └── attack_mapping.yaml
├── src/
│   ├── preprocessing.py
│   ├── state_builder.py
│   ├── scaling.py
│   ├── world_model.py
│   ├── attack_mapping.py
│   ├── explainability.py
│   └── pcap_ingest.py
├── training/
│   └── train.py
└── models/
    ├── world_model.pt
    ├── benchmark_results.json
    └── train_log.txt
```

The main areas are `src/` for preprocessing and model components, `training/` for model fitting and evaluation, `configs/` for settings and attack mappings, and `models/` for saved weights and benchmark artifacts.

---

## Data-quality handling

The preprocessing pipeline addresses two issues noted in the source project:

1. **Timestamps without AM/PM markers:** Some CIC-IDS2018 timestamps use a 12-hour clock without an AM/PM indicator. If interpreted as 24-hour timestamps without correction, afternoon traffic may be placed before morning traffic. The preprocessing code shifts hours below 8 by 12 when the file appears to follow this timestamp pattern.
2. **Epoch-zero timestamps:** A small number of rows may contain timestamps from 1970. At 10-second resolution, these can create an enormous number of empty time windows and consume excessive memory. Rows more than two days from the median timestamp are removed, and the state builder checks for an excessive number of windows.

These checks are important because incorrect time ordering or invalid timestamps can distort the temporal sequences used for forecasting.

---

## Limitations and considerations

- A time window is labelled hostile if at least one flow in it is hostile. Mixed benign/attack windows are therefore labelled entirely as attack.
- Attacks in IDS2018 can last for hours, so next-window and multi-step metrics may reflect attack persistence rather than early prediction. The early-warning metric is intended to isolate the more difficult onset-prediction setting.
- Forecast lead time is limited to 60 seconds by design; the results do not establish prediction at longer horizons.
- Source and destination IP addresses are excluded to reduce memorization of specific hosts. As a consequence, host reputation is not used.
- Evaluation is based on a single dataset. The test split assesses generalization to attack families withheld during training, not generalization to entirely different networks.
- **ATT&CK stage prediction is experimental.** The model can only predict tactics represented during training. Since Infiltration and Bot are absent from the training days, test-split stage accuracy is zero by construction. The source project also reports low validation accuracy (0.053), so this output should not be treated as a reliable classification result.
- The training script reads at most the first 600,000 rows from each CSV, so not every flow in every capture is necessarily included.

---

## Attribution

This repository adapts the implementation and technical approach documented in the [upstream project repository](https://github.com/Nikethan10/trinetra-sih2026-ps26153). The README has been reorganized for this project; implementation details and benchmark values should be treated as inherited from the upstream work unless independently changed and verified.

Before redistributing, review the upstream license and preserve any required copyright, license, and attribution notices.

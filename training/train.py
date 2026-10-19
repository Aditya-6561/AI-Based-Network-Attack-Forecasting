"""
Training and benchmarking pipeline for the TRINETRA temporal world model.

Protocol
--------
Split is grouped by capture day. Every test day is strictly later than every
train and validation day, and the test attack families never appear in either:

  train : 14/02, 15/02, 21/02, 22/02   (brute force, DoS, DDoS, web)
  val   : 16/02, 23/02                 (DoS Hulk/SlowHTTPTest, web)
                                       -> early stopping + threshold selection
  test  : 28/02, 01/03, 02/03          (Infiltration, Bot) -> reported once

Every reported number is produced by this script and written to
models/benchmark_results.json. Nothing in the dashboard is hardcoded.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attack_mapping import AttackMapper                            # noqa: E402
from src.preprocessing import REQUIRED_COLUMNS, TelemetryPreprocessor  # noqa: E402
from src.scaling import LogStandardScaler                              # noqa: E402
from src.state_builder import STATE_DIM, StateBuilder                  # noqa: E402
from src.world_model import (NUM_STAGES, TemporalWorldModel,           # noqa: E402
                             VanillaLSTMClassifier)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "models", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# Point TRINETRA_DATA_DIR at the folder holding the IDS2018 CSVs, or pass --data-dir.
DEFAULT_DATA_DIR = os.environ.get("TRINETRA_DATA_DIR", os.path.join(ROOT, "data"))

# Validation must contain attacks with enough volumetric presence to select a
# model and a threshold. The web-attack days (22-23 Feb) carry only a few hundred
# hostile flows in 600k, so a validation split built from them alone has almost no
# signal at 10s granularity and early stopping degenerates to picking an
# undertrained epoch. 16 Feb (DoS Hulk / SlowHTTPTest) is held out instead.
# The test days remain strictly later than every train and validation day, and
# their attack families (Infiltration, Bot) appear in neither.
TRAIN_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
]
VAL_FILES = [
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
]
TEST_FILES = [
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

WINDOW_SEC = 10
HISTORY_STEPS = 12
FORECAST_STEPS = 6
SEED = 1337


# --------------------------------------------------------------------------- #
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(file_list, data_dir, max_rows, stride):
    """
    Returns X, Y_state, Y_risk, Y_stage and a per-sequence flag saying whether the
    OBSERVED history already contained an attack. That flag is what lets us
    separate genuine forecasting from persistence of an ongoing attack.
    """
    pre = TelemetryPreprocessor(anonymize_ips=True)
    sb = StateBuilder(window_duration_sec=WINDOW_SEC, history_steps=HISTORY_STEPS,
                      forecast_steps=FORECAST_STEPS, stride=stride, mapper=AttackMapper())

    Xs, Ys, Yr, Yg, HistAtk = [], [], [], [], []
    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print("   WARNING: missing %s -- skipping" % fname)
            continue

        t0 = time.time()
        cache = os.path.join(CACHE_DIR, "%s_r%d_w%d_h%d_k%d_s%d.npz"
                             % (os.path.splitext(fname)[0], max_rows, WINDOW_SEC,
                                HISTORY_STEPS, FORECAST_STEPS, stride))
        if os.path.exists(cache):
            z = np.load(cache)
            X, Y_state, Y_risk = z["X"], z["Y_state"], z["Y_risk"]
            Y_stage, hist_attack = z["Y_stage"], z["hist_attack"]
            note = "cached"
        else:
            df = pre.load_csv_file(fpath, nrows=max_rows, usecols=REQUIRED_COLUMNS)
            matrix, _, is_attack, _ = sb.build_state_matrix(df)
            X, Y_state, Y_risk, Y_stage, _ = sb.generate_sequences(df)
            if len(X) == 0:
                print("   WARNING: %s produced no sequences" % fname)
                continue

            W, K = HISTORY_STEPS, FORECAST_STEPS
            starts = np.arange(0, len(matrix) - (W + K) + 1, sb.stride)
            hist_idx = starts[:, None] + np.arange(W)[None, :]
            hist_attack = is_attack[hist_idx].max(axis=1)

            np.savez_compressed(cache, X=X, Y_state=Y_state, Y_risk=Y_risk,
                                Y_stage=Y_stage, hist_attack=hist_attack)
            note = "built"

        Xs.append(X); Ys.append(Y_state); Yr.append(Y_risk)
        Yg.append(Y_stage); HistAtk.append(hist_attack)
        print("   %-52s %6d seqs  attack=%.3f  (%s, %.1fs)"
              % (fname[:52], len(X), float(Y_risk.mean()), note, time.time() - t0))

    if not Xs:
        raise RuntimeError("No sequences generated -- check --data-dir / TRINETRA_DATA_DIR.")

    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Yr),
            np.concatenate(Yg), np.concatenate(HistAtk))


# --------------------------------------------------------------------------- #
def binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        out["fpr"] = float(fp / max(fp + tn, 1))
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out.update({"fpr": 0.0, "auroc": 0.5, "auprc": out["positive_rate"]})
    return out


def pick_threshold(y_true, y_prob):
    """Choose the F1-optimal threshold on VALIDATION data only."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99))):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def pick_threshold_at_fpr(y_true, y_prob, target_fpr):
    """
    Lowest threshold whose false-positive rate on VALIDATION stays within budget.

    Comparing models at an F1-optimal threshold is fragile here: the risk-score
    distribution shifts between attack families, so a threshold tuned on DoS days
    transfers badly to Infiltration/Bot and the comparison ends up measuring
    calibration drift rather than discrimination. Holding every model to the same
    false-alarm budget is how a SOC actually tunes a detector, and it makes the
    comparison fair.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    neg = y_prob[y_true == 0]
    if neg.size == 0:
        return 0.5
    return float(np.quantile(neg, 1.0 - target_fpr))


def early_warning_metrics(y_risk, hist_attack, y_prob, threshold, window_sec=WINDOW_SEC):
    """
    Genuine forecasting evaluation.

    Restricted to ONSET sequences: no attack anywhere in the observed history, but
    an attack appears inside the K-step forecast horizon. Persistence of an
    already-visible attack cannot score here, so this isolates prediction from
    detection.

    Lead time = (index of first future attack window + 1) * window_sec, i.e. how
    far ahead of the actual onset the alert fires. Bounded above by K * window_sec.
    """
    y_risk = np.asarray(y_risk)
    hist_attack = np.asarray(hist_attack).astype(bool)
    y_prob = np.asarray(y_prob)

    future_attack = y_risk.max(axis=1) > 0.5
    onset = future_attack & (~hist_attack)
    quiet = (~future_attack) & (~hist_attack)

    n_onset = int(onset.sum())
    if n_onset == 0:
        return {"n_onset_sequences": 0, "early_warning_recall": 0.0,
                "mean_lead_time_sec": 0.0, "onset_false_alarm_rate": 0.0}

    alerted = y_prob[onset].max(axis=1) >= threshold
    first_attack_step = np.argmax(y_risk[onset] > 0.5, axis=1)
    lead = (first_attack_step[alerted] + 1) * window_sec

    false_alarm = 0.0
    if quiet.sum() > 0:
        false_alarm = float((y_prob[quiet].max(axis=1) >= threshold).mean())

    return {
        "n_onset_sequences": n_onset,
        "early_warning_recall": float(alerted.mean()),
        "mean_lead_time_sec": float(lead.mean()) if alerted.any() else 0.0,
        "onset_false_alarm_rate": false_alarm,
    }


# --------------------------------------------------------------------------- #
def train_world_model(Xtr, Ystr, Yrtr, Ygtr, Xva, Yrva, device, epochs, batch_size,
                      lr, alpha, beta, gamma, latent_dim=48, hidden_dim=96):
    model = TemporalWorldModel(input_dim=STATE_DIM, hidden_dim=hidden_dim,
                               latent_dim=latent_dim, forecast_steps=FORECAST_STEPS,
                               num_stages=NUM_STAGES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Validation AUPRC swung between 0.20 and 0.66 across consecutive epochs at a
    # flat learning rate, which makes early stopping pick a lucky epoch rather than
    # a good model. Cosine decay settles the late epochs so the selection means
    # something.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)

    pos = float(Yrtr.sum())
    neg = float(Yrtr.size - pos)
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], device=device)
    print("   risk pos_weight = %.2f (train positive rate %.4f)"
          % (pos_weight.item(), pos / Yrtr.size))

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    # Stage 0 (benign) is ~90% of windows. With unweighted cross-entropy the stage
    # head simply always answers "benign" and scores ~0 accuracy on the windows
    # that actually matter. Inverse-frequency weights make the head predict a
    # tactic instead of the prior.
    counts = np.bincount(Ygtr.reshape(-1), minlength=NUM_STAGES).astype(np.float64)
    stage_w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 0.0)
    stage_w = stage_w / stage_w[stage_w > 0].mean()
    print("   stage class weights = %s" % np.round(stage_w, 2).tolist())
    ce = nn.CrossEntropyLoss(
        weight=torch.tensor(stage_w, dtype=torch.float32, device=device))

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Ystr),
                       torch.tensor(Yrtr), torch.tensor(Ygtr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    t_Xva = torch.tensor(Xva, device=device)
    best_auprc, best_state, best_epoch = -1.0, None, -1

    for epoch in range(1, epochs + 1):
        model.train()
        tot = tot_s = tot_r = tot_g = 0.0
        for bx, bys, byr, byg in loader:
            bx, bys = bx.to(device), bys.to(device)
            byr, byg = byr.to(device), byg.to(device)

            opt.zero_grad()
            pred_states, risk_logits, stage_logits = model(bx)

            l_state = mse(pred_states, bys)
            l_risk = bce(risk_logits, byr)
            l_stage = ce(stage_logits.reshape(-1, NUM_STAGES), byg.reshape(-1))

            loss = alpha * l_state + beta * l_risk + gamma * l_stage
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += loss.item(); tot_s += l_state.item()
            tot_r += l_risk.item(); tot_g += l_stage.item()

        n = len(loader)
        model.eval()
        with torch.no_grad():
            _, va_logits, _ = model(t_Xva)
            va_prob = torch.sigmoid(va_logits).cpu().numpy()
        auprc = (average_precision_score(Yrva.ravel().astype(int), va_prob.ravel())
                 if len(np.unique(Yrva)) > 1 else 0.0)

        sched.step()
        flag = ""
        if auprc > best_auprc:
            best_auprc, best_epoch = auprc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print("   epoch %2d/%d | loss %.4f (state %.4f | risk %.4f | stage %.4f)"
              " | val AUPRC %.4f%s"
              % (epoch, epochs, tot / n, tot_s / n, tot_r / n, tot_g / n, auprc, flag))

    if best_state is not None:
        model.load_state_dict(best_state)
    print("   restored best epoch %d (val AUPRC %.4f)" % (best_epoch, best_auprc))
    return model, best_epoch, best_auprc


def train_lstm_baseline(Xtr, Yrtr, device, epochs, batch_size, lr):
    model = VanillaLSTMClassifier(input_dim=STATE_DIM, hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y = Yrtr[:, 0]
    pos = float(y.sum()); neg = float(y.size - pos)
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-rows", type=int, default=600000,
                    help="rows read per CSV (600k spans a full capture day)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5, help="state reconstruction weight")
    ap.add_argument("--beta", type=float, default=3.0, help="risk weight")
    ap.add_argument("--gamma", type=float, default=1.0, help="ATT&CK stage weight")
    ap.add_argument("--fpr-budget", type=float, default=0.05,
                    help="validation false-alarm budget for the equal-FPR operating point")
    ap.add_argument("--latent-dim", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=96)
    args = ap.parse_args()

    set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("TRINETRA -- temporal world model training & benchmark")
    print("=" * 78)
    print("device: %s | rows/file: %d | stride: %d" % (device, args.max_rows, args.stride))

    print("\n[1/6] Loading TRAIN split (14/02, 15/02, 21/02, 22/02)")
    Xtr, Ystr, Yrtr, Ygtr, Htr = load_split(TRAIN_FILES, args.data_dir, args.max_rows, args.stride)
    print("\n[2/6] Loading VAL split (16/02, 23/02)")
    Xva, Ysva, Yrva, Ygva, Hva = load_split(VAL_FILES, args.data_dir, args.max_rows, args.stride)
    print("\n[3/6] Loading TEST split (28/02 - 02/03, unseen attack families)")
    Xte, Yste, Yrte, Ygte, Hte = load_split(TEST_FILES, args.data_dir, args.max_rows, args.stride)

    print("\n   shapes  train %s | val %s | test %s" % (Xtr.shape, Xva.shape, Xte.shape))
    print("   attack rate  train %.4f | val %.4f | test %.4f"
          % (Yrtr.mean(), Yrva.mean(), Yrte.mean()))

    # ---- normalisation: fit on TRAIN ONLY --------------------------------- #
    scaler = LogStandardScaler().fit(Xtr.reshape(-1, STATE_DIM))

    def nz(A):
        return scaler.transform(A.reshape(-1, STATE_DIM)).reshape(A.shape)

    Xtr_n, Xva_n, Xte_n = nz(Xtr), nz(Xva), nz(Xte)
    Ystr_n = nz(Ystr)

    # ---------------------------------------------------------------- #
    print("\n[4/6] Baseline 1: logistic regression on the current state S_t")
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_clf.fit(Xtr_n[:, -1, :], Yrtr[:, 0])
    lr_va = lr_clf.predict_proba(Xva_n[:, -1, :])[:, 1]
    lr_te = lr_clf.predict_proba(Xte_n[:, -1, :])[:, 1]
    # A static classifier has no horizon concept: it emits one score for the whole
    # forecast window. Broadcasting keeps the K-horizon comparison apples-to-apples.
    lr_te_K = np.repeat(lr_te[:, None], FORECAST_STEPS, axis=1)
    thr_lr = pick_threshold(Yrva[:, 0], lr_va)

    print("\n[5/6] Baseline 2: vanilla LSTM sequence classifier")
    lstm = train_lstm_baseline(Xtr_n, Yrtr, device, epochs=8,
                               batch_size=args.batch_size, lr=args.lr)
    with torch.no_grad():
        ls_va = torch.sigmoid(lstm(torch.tensor(Xva_n, device=device))).cpu().numpy()
        ls_te = torch.sigmoid(lstm(torch.tensor(Xte_n, device=device))).cpu().numpy()
    ls_te_K = np.repeat(ls_te[:, None], FORECAST_STEPS, axis=1)
    thr_ls = pick_threshold(Yrva[:, 0], ls_va)

    print("\n[6/6] Proposed: temporal latent state-transition world model")
    wm, best_epoch, best_auprc = train_world_model(
        Xtr_n, Ystr_n, Yrtr, Ygtr, Xva_n, Yrva,
        device, args.epochs, args.batch_size, args.lr, args.alpha, args.beta, args.gamma,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)

    with torch.no_grad():
        _, wm_va_log, _ = wm(torch.tensor(Xva_n, device=device))
        wm_va = torch.sigmoid(wm_va_log).cpu().numpy()
        _, wm_te_log, wm_te_stage = wm(torch.tensor(Xte_n, device=device))
        wm_te = torch.sigmoid(wm_te_log).cpu().numpy()
        stage_pred_te = wm_te_stage.argmax(-1).cpu().numpy()
    thr_wm = pick_threshold(Yrva, wm_va)

    # ---------------------------------------------------------------- #
    results = {
        "protocol": {
            "split": "chronological by capture day",
            "train_files": TRAIN_FILES, "val_files": VAL_FILES, "test_files": TEST_FILES,
            "window_sec": WINDOW_SEC, "history_steps": HISTORY_STEPS,
            "forecast_steps": FORECAST_STEPS, "rows_per_file": args.max_rows,
            "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
            "test_positive_rate": float(Yrte.mean()),
            "thresholds_selected_on": "validation split",
            "best_epoch": int(best_epoch), "best_val_auprc": float(best_auprc),
            "seed": SEED,
        },
        "models": {},
    }

    # A constant "always alert" predictor. Included because on a split with a 38%
    # positive rate it already scores F1 ~= 0.55; any model not clearly above it is
    # not doing useful work, and reporting it makes that impossible to hide.
    ones_t1 = np.ones_like(Yrte[:, 0])
    ones_K = np.ones_like(Yrte)

    specs = [
        ("Always-alert (trivial reference)", ones_t1, ones_K, 0.5, None),
        ("Logistic Regression (static S_t)", lr_te, lr_te_K, thr_lr, lr_va),
        ("Vanilla LSTM classifier", ls_te, ls_te_K, thr_ls, ls_va),
        ("Temporal World Model (K-step rollout)", wm_te[:, 0], wm_te, thr_wm, wm_va),
    ]
    results["protocol"]["equal_fpr_budget"] = args.fpr_budget

    for name, p_t1, p_K, thr, p_va in specs:
        entry = {
            "next_window": binary_metrics(Yrte[:, 0], p_t1, thr),
            "k_horizon": binary_metrics(Yrte, p_K, thr),
            "early_warning": early_warning_metrics(Yrte, Hte, p_K, thr),
        }
        # Second operating point: every model held to the same validation
        # false-alarm budget, so the comparison is not decided by how well each
        # model's threshold happens to transfer across attack families.
        if p_va is None:
            thr_b = 0.5
        else:
            # A per-horizon score is matched against per-horizon labels; a static
            # score against the t+1 labels it was fitted on.
            y_va = Yrva if p_va.ndim > 1 else Yrva[:, 0]
            thr_b = pick_threshold_at_fpr(y_va, p_va, args.fpr_budget)
        entry["k_horizon_equal_fpr"] = binary_metrics(Yrte, p_K, thr_b)
        entry["early_warning_equal_fpr"] = early_warning_metrics(Yrte, Hte, p_K, thr_b)
        results["models"][name] = entry

    with torch.no_grad():
        _, _, va_stage = wm(torch.tensor(Xva_n, device=device))
        stage_pred_va = va_stage.argmax(-1).cpu().numpy()

    atk_te = Yrte > 0.5
    atk_va = Yrva > 0.5
    results["attack_stage"] = {
        # Test families (Infiltration, Bot) never appear in training, so the stage
        # head cannot emit their indices: test stage accuracy is 0 by construction.
        # Validation accuracy is the meaningful number for this head.
        "val_accuracy_on_attack_windows":
            float((stage_pred_va[atk_va] == Ygva[atk_va]).mean()) if atk_va.sum() else 0.0,
        "test_accuracy_on_attack_windows":
            float((stage_pred_te[atk_te] == Ygte[atk_te]).mean()) if atk_te.sum() else 0.0,
        "test_families_unseen_in_training": True,
        "stages_present_in_train": sorted(set(int(v) for v in np.unique(Ygtr))),
        "stages_present_in_test": sorted(set(int(v) for v in np.unique(Ygte))),
    }

    torch.save({
        "model_state": wm.state_dict(),
        "scaler_state": scaler.state_dict(),
        "input_dim": STATE_DIM, "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "forecast_steps": FORECAST_STEPS, "history_steps": HISTORY_STEPS,
        "window_sec": WINDOW_SEC, "num_stages": NUM_STAGES,
        "alert_threshold": float(thr_wm),
    }, os.path.join(MODEL_DIR, "world_model.pt"))

    with open(os.path.join(MODEL_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("TEST-SET RESULTS  (28/02 - 02/03; Infiltration + Bot unseen in training)")
    print("=" * 78)
    hdr = "%-38s %6s %6s %6s %6s %7s" % ("model", "F1", "Prec", "Rec", "FPR", "AUPRC")
    for task in ("next_window", "k_horizon"):
        title = "t+1 detection" if task == "next_window" else "K=6 horizon forecast (60 s)"
        print("\n-- %s --" % title)
        print(hdr)
        for name in results["models"]:
            m = results["models"][name][task]
            print("%-38s %6.4f %6.4f %6.4f %6.4f %7.4f"
                  % (name[:38], m["f1"], m["precision"], m["recall"], m["fpr"], m["auprc"]))

    print("\n-- early warning (benign history -> attack inside horizon) --")
    print("%-38s %8s %10s %12s" % ("model", "recall", "lead(s)", "false_alarm"))
    for name in results["models"]:
        e = results["models"][name]["early_warning"]
        print("%-38s %8.4f %10.1f %12.4f"
              % (name[:38], e["early_warning_recall"], e["mean_lead_time_sec"],
                 e["onset_false_alarm_rate"]))
    print("   (%d onset sequences in test split)"
          % results["models"][specs[0][0]]["early_warning"]["n_onset_sequences"])

    print("\n== equal false-alarm budget (%.0f%% FPR fixed on validation) =="
          % (100 * args.fpr_budget))
    print("\n-- K=6 horizon forecast --")
    print(hdr)
    for name in results["models"]:
        m = results["models"][name]["k_horizon_equal_fpr"]
        print("%-38s %6.4f %6.4f %6.4f %6.4f %7.4f"
              % (name[:38], m["f1"], m["precision"], m["recall"], m["fpr"], m["auprc"]))
    print("\n-- early warning --")
    print("%-38s %8s %10s %12s" % ("model", "recall", "lead(s)", "false_alarm"))
    for name in results["models"]:
        e = results["models"][name]["early_warning_equal_fpr"]
        print("%-38s %8.4f %10.1f %12.4f"
              % (name[:38], e["early_warning_recall"], e["mean_lead_time_sec"],
                 e["onset_false_alarm_rate"]))

    st = results["attack_stage"]
    print("\n-- ATT&CK stage prediction --")
    print("   validation accuracy on attack windows: %.4f  (stages seen in training)"
          % st["val_accuracy_on_attack_windows"])
    print("   test accuracy on attack windows:       %.4f"
          % st["test_accuracy_on_attack_windows"])
    print("   train stages %s vs test stages %s -- unseen families cannot be named"
          % (st["stages_present_in_train"], st["stages_present_in_test"]))

    print("\nSaved: models/world_model.pt and models/benchmark_results.json")
    print("=" * 78)


if __name__ == "__main__":
    main()

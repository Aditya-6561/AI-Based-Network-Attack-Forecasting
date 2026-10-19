import numpy as np
import pandas as pd

STATE_DIM = 42

# Upper bound on windows per capture; a tripwire for corrupt timestamps.
MAX_WINDOWS = 2_000_000

# Canonical ordering of the network-state vector S_t. Index i here is index i in
# src.explainability.FEATURE_NAMES -- keep the two lists in lockstep.
FEATURE_ORDER = [
    "flow_count", "tot_fwd_pkts", "tot_bwd_pkts", "tot_fwd_bytes", "tot_bwd_bytes",
    "flow_duration_mean", "flow_duration_std", "bytes_per_sec", "pkts_per_sec",
    "syn_cnt", "ack_cnt", "rst_cnt", "fin_cnt", "psh_cnt", "urg_cnt", "syn_ack_ratio",
    "iat_mean", "iat_std", "iat_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "init_fwd_win", "init_bwd_win", "fwd_seg_min", "dst_port_entropy",
    "proto_tcp_ratio", "proto_udp_ratio", "active_mean", "idle_mean",
    "subflow_fwd_pkts", "subflow_bwd_pkts", "subflow_fwd_byts", "subflow_bwd_byts",
    "down_up_ratio", "fwd_pkts_per_sec", "bwd_pkts_per_sec", "fwd_iat_mean",
    "bwd_iat_mean", "pkt_size_avg", "uniq_dst_ports", "silent_flag",
]
assert len(FEATURE_ORDER) == STATE_DIM


class StateBuilder:
    """
    Vectorised network-state aggregator.

    Resamples flow records into fixed-duration windows and emits, per window, a
    42-dimensional state vector S_t combining flow-level aggregates (volumes, TCP
    flag distributions, IAT statistics) with packet-level descriptors (TCP initial
    window size, segment size floor, payload size distribution, destination-port
    entropy and cardinality as port-scan signatures).

    Emits supervised sequence tensors:
      X       (N, W, D)  observed history S_{t-W:t}
      Y_state (N, K, D)  ground-truth future states S_{t+1:t+K}
      Y_risk  (N, K)     binary attack indicator per future window
      Y_stage (N, K)     ATT&CK stage index per future window
    """

    def __init__(self,
                 window_duration_sec: int = 10,
                 history_steps: int = 12,
                 forecast_steps: int = 6,
                 stride: int = 2,
                 mapper=None):
        self.window_duration_sec = window_duration_sec
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.stride = max(1, int(stride))
        self.mapper = mapper

    # ------------------------------------------------------------------ #
    @staticmethod
    def _agg(res, df, col, how, default_index):
        """Aggregate `col` with `how`, or return zeros when the column is absent."""
        if col not in df.columns:
            return pd.Series(0.0, index=default_index)
        out = getattr(res[col], how)()
        return out.reindex(default_index).fillna(0.0)

    @staticmethod
    def _port_entropy_and_cardinality(ports, bucket, index):
        """
        Shannon entropy H(dst_port) and unique-port count per window.

        High entropy with high cardinality over a short window is the canonical
        horizontal port-scan signature; low entropy with high volume indicates a
        flood against a single service.
        """
        if ports is None:
            zeros = pd.Series(0.0, index=index)
            return zeros, zeros.copy()

        pair = pd.DataFrame({"b": np.asarray(bucket), "p": np.asarray(ports)})
        counts = pair.groupby(["b", "p"], sort=False).size().rename("n").reset_index()
        totals = counts.groupby("b")["n"].transform("sum")
        p = counts["n"] / totals
        counts["h"] = -p * np.log2(p)

        entropy = counts.groupby("b")["h"].sum().reindex(index).fillna(0.0)
        cardinality = counts.groupby("b").size().reindex(index).fillna(0.0)
        return entropy, cardinality

    # ------------------------------------------------------------------ #
    def build_state_matrix(self, df: pd.DataFrame):
        """Returns (state_matrix (T, D), window_timestamps, is_attack (T,), stage (T,))."""
        if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
            raise ValueError("A valid 'Timestamp' column is required to build network state.")

        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
        rule = str(self.window_duration_sec) + "s"

        # Guard against corrupt timestamps producing an absurd number of empty
        # buckets (a single 1970 epoch row spans ~48 years at 10s resolution).
        span_sec = (df.index.max() - df.index.min()).total_seconds()
        n_buckets = span_sec / self.window_duration_sec
        if n_buckets > MAX_WINDOWS:
            raise ValueError(
                "Capture spans %.1f days -> %.0f windows at %ds resolution, which "
                "exceeds the %d-window limit. This usually means corrupt timestamps "
                "survived preprocessing." % (span_sec / 86400, n_buckets,
                                             self.window_duration_sec, MAX_WINDOWS))

        res = df.resample(rule)

        flow_counts = res.size()
        idx = flow_counts.index
        if len(flow_counts) == 0:
            empty = np.empty((0, STATE_DIM), dtype=np.float32)
            return empty, idx, np.empty(0), np.empty(0)

        def A(col, how):
            return self._agg(res, df, col, how, idx)

        syn = A("SYN Flag Cnt", "sum")
        ack = A("ACK Flag Cnt", "sum")

        bucket = df.index.floor(rule)
        ports = df["Dst Port"].to_numpy() if "Dst Port" in df.columns else None
        port_entropy, port_card = self._port_entropy_and_cardinality(ports, bucket, idx)

        if "Protocol" in df.columns:
            proto = df["Protocol"]
            tcp_ratio = (proto == 6).groupby(bucket).mean().reindex(idx).fillna(0.0)
            udp_ratio = (proto == 17).groupby(bucket).mean().reindex(idx).fillna(0.0)
        else:
            tcp_ratio = pd.Series(0.0, index=idx)
            udp_ratio = pd.Series(0.0, index=idx)

        cols = {
            "flow_count": flow_counts,
            "tot_fwd_pkts": A("Tot Fwd Pkts", "sum"),
            "tot_bwd_pkts": A("Tot Bwd Pkts", "sum"),
            "tot_fwd_bytes": A("TotLen Fwd Pkts", "sum"),
            "tot_bwd_bytes": A("TotLen Bwd Pkts", "sum"),
            "flow_duration_mean": A("Flow Duration", "mean"),
            "flow_duration_std": A("Flow Duration", "std"),
            "bytes_per_sec": A("Flow Byts/s", "mean"),
            "pkts_per_sec": A("Flow Pkts/s", "mean"),
            "syn_cnt": syn,
            "ack_cnt": ack,
            "rst_cnt": A("RST Flag Cnt", "sum"),
            "fin_cnt": A("FIN Flag Cnt", "sum"),
            "psh_cnt": A("PSH Flag Cnt", "sum"),
            "urg_cnt": A("URG Flag Cnt", "sum"),
            "syn_ack_ratio": (syn + 1.0) / (ack + 1.0),
            "iat_mean": A("Flow IAT Mean", "mean"),
            "iat_std": A("Flow IAT Std", "mean"),
            "iat_max": A("Flow IAT Max", "max"),
            "pkt_len_mean": A("Pkt Len Mean", "mean"),
            "pkt_len_std": A("Pkt Len Std", "mean"),
            "pkt_len_var": A("Pkt Len Var", "mean"),
            "init_fwd_win": A("Init Fwd Win Byts", "mean"),
            "init_bwd_win": A("Init Bwd Win Byts", "mean"),
            "fwd_seg_min": A("Fwd Seg Size Min", "mean"),
            "dst_port_entropy": port_entropy,
            "proto_tcp_ratio": tcp_ratio,
            "proto_udp_ratio": udp_ratio,
            "active_mean": A("Active Mean", "mean"),
            "idle_mean": A("Idle Mean", "mean"),
            "subflow_fwd_pkts": A("Subflow Fwd Pkts", "sum"),
            "subflow_bwd_pkts": A("Subflow Bwd Pkts", "sum"),
            "subflow_fwd_byts": A("Subflow Fwd Byts", "sum"),
            "subflow_bwd_byts": A("Subflow Bwd Byts", "sum"),
            "down_up_ratio": A("Down/Up Ratio", "mean"),
            "fwd_pkts_per_sec": A("Fwd Pkts/s", "mean"),
            "bwd_pkts_per_sec": A("Bwd Pkts/s", "mean"),
            "fwd_iat_mean": A("Fwd IAT Mean", "mean"),
            "bwd_iat_mean": A("Bwd IAT Mean", "mean"),
            "pkt_size_avg": A("Pkt Size Avg", "mean"),
            "uniq_dst_ports": port_card,
            "silent_flag": pd.Series((flow_counts.values == 0).astype(float), index=idx),
        }

        matrix = np.column_stack([np.asarray(cols[name], dtype=np.float64)
                                  for name in FEATURE_ORDER])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # ---- labels -------------------------------------------------- #
        if "Label" in df.columns:
            lab = df["Label"].astype(str).str.strip()
            is_atk_flow = (lab.str.lower() != "benign")
            is_attack = is_atk_flow.astype(np.float32).groupby(bucket).max()
            is_attack = is_attack.reindex(idx).fillna(0.0).to_numpy().astype(np.float32)

            if self.mapper is not None:
                lut = {s: self.mapper.map_label(s)[2] for s in lab.unique()}
                stage_per_flow = lab.map(lut).astype(np.int64)
                stage = stage_per_flow.groupby(bucket).max().reindex(idx).fillna(0)
                stage = stage.to_numpy().astype(np.int64)
            else:
                stage = is_attack.astype(np.int64)
        else:
            is_attack = np.zeros(len(idx), dtype=np.float32)
            stage = np.zeros(len(idx), dtype=np.int64)

        return matrix, idx, is_attack, stage

    # ------------------------------------------------------------------ #
    def generate_sequences(self, df: pd.DataFrame):
        """Returns X, Y_state, Y_risk, Y_stage, anchor_timestamps."""
        matrix, idx, is_attack, stage = self.build_state_matrix(df)

        W, K = self.history_steps, self.forecast_steps
        req = W + K
        T = len(matrix)

        if T < req:
            return (np.empty((0, W, STATE_DIM), np.float32),
                    np.empty((0, K, STATE_DIM), np.float32),
                    np.empty((0, K), np.float32),
                    np.empty((0, K), np.int64),
                    pd.DatetimeIndex([]))

        starts = np.arange(0, T - req + 1, self.stride)
        hist_idx = starts[:, None] + np.arange(W)[None, :]
        fut_idx = starts[:, None] + np.arange(W, req)[None, :]

        X = matrix[hist_idx]
        Y_state = matrix[fut_idx]
        Y_risk = is_attack[fut_idx].astype(np.float32)
        Y_stage = stage[fut_idx].astype(np.int64)
        anchors = idx[starts + W - 1]

        return X, Y_state, Y_risk, Y_stage, anchors

"""
PCAP -> CICFlowMeter-style flow records.

The world model consumes windowed flow statistics. This module reconstructs those
statistics directly from raw packets so the system can ingest a .pcap/.pcapng
capture as well as an IDS2018 CSV, and so packet-level attributes the CSVs never
expose (TTL variance, TCP window size evolution, IP fragmentation, retransmission
counts) can be measured at their source.

Flows are keyed on the canonical bidirectional 5-tuple; the direction first seen
is treated as forward. Scapy is imported lazily so the rest of the project runs
without it installed.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


class _Flow:
    __slots__ = ("src", "dst", "sport", "dport", "proto", "start", "last",
                 "f_times", "b_times", "f_len", "b_len", "flags",
                 "ttl_f", "win_f", "win_b", "frag", "seqs_seen", "retrans",
                 "init_win_f", "init_win_b", "seg_min_f")

    def __init__(self, src, dst, sport, dport, proto, t):
        self.src, self.dst = src, dst
        self.sport, self.dport, self.proto = sport, dport, proto
        self.start = self.last = t
        self.f_times, self.b_times = [], []
        self.f_len, self.b_len = [], []
        self.flags = defaultdict(int)
        self.ttl_f, self.win_f, self.win_b = [], [], []
        self.frag = 0
        self.seqs_seen = set()
        self.retrans = 0
        self.init_win_f = self.init_win_b = 0
        self.seg_min_f = None


def _stats(values):
    """(mean, std, max, min) with empty-safe defaults."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def _iats(times):
    """Inter-arrival times in microseconds, matching the CICFlowMeter unit."""
    if len(times) < 2:
        return []
    return list(np.diff(np.asarray(times, dtype=np.float64)) * 1e6)


def parse_pcap(path, max_packets=300000, flow_timeout=120.0):
    """
    Parse a capture into a DataFrame with IDS2018-compatible column names.

    Returns a DataFrame carrying the columns in
    src.preprocessing.REQUIRED_COLUMNS (minus Label, which is set to 'Benign'
    because a raw capture is unlabelled), plus packet-level extras.
    """
    try:
        from scapy.all import IP, TCP, UDP, PcapReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PCAP ingestion needs scapy. Install it with:  pip install scapy"
        ) from exc

    flows = {}
    n = 0

    with PcapReader(path) as reader:
        for pkt in reader:
            if n >= max_packets:
                break
            if IP not in pkt:
                continue
            n += 1

            ip = pkt[IP]
            t = float(pkt.time)
            proto = int(ip.proto)

            if TCP in pkt:
                sport, dport, layer = int(pkt[TCP].sport), int(pkt[TCP].dport), pkt[TCP]
            elif UDP in pkt:
                sport, dport, layer = int(pkt[UDP].sport), int(pkt[UDP].dport), None
            else:
                sport = dport = 0
                layer = None

            fwd_key = (ip.src, ip.dst, sport, dport, proto)
            bwd_key = (ip.dst, ip.src, dport, sport, proto)

            if fwd_key in flows and t - flows[fwd_key].last <= flow_timeout:
                flow, forward = flows[fwd_key], True
            elif bwd_key in flows and t - flows[bwd_key].last <= flow_timeout:
                flow, forward = flows[bwd_key], False
            else:
                flow = _Flow(ip.src, ip.dst, sport, dport, proto, t)
                flows[fwd_key] = flow
                forward = True

            flow.last = t
            plen = len(ip.payload) if layer is None else len(layer.payload)

            if forward:
                flow.f_times.append(t)
                flow.f_len.append(plen)
                flow.ttl_f.append(int(ip.ttl))
            else:
                flow.b_times.append(t)
                flow.b_len.append(plen)

            # IP fragmentation: MF bit set or a non-zero fragment offset.
            if (int(ip.flags) & 0x1) or int(ip.frag) != 0:
                flow.frag += 1

            if layer is not None:
                f = int(layer.flags)
                for bit, name in ((FIN, "fin"), (SYN, "syn"), (RST, "rst"),
                                  (PSH, "psh"), (ACK, "ack"), (URG, "urg")):
                    if f & bit:
                        flow.flags[name] += 1

                win = int(layer.window)
                if forward:
                    flow.win_f.append(win)
                    if not flow.init_win_f:
                        flow.init_win_f = win
                    hdr = layer.dataofs * 4 if layer.dataofs else 20
                    flow.seg_min_f = hdr if flow.seg_min_f is None else min(flow.seg_min_f, hdr)
                else:
                    flow.win_b.append(win)
                    if not flow.init_win_b:
                        flow.init_win_b = win

                # Retransmission heuristic: a sequence number carrying payload
                # that we have already recorded for this direction.
                if plen > 0:
                    key = (forward, int(layer.seq))
                    if key in flow.seqs_seen:
                        flow.retrans += 1
                    else:
                        flow.seqs_seen.add(key)

    rows = []
    for flow in flows.values():
        all_times = sorted(flow.f_times + flow.b_times)
        all_len = flow.f_len + flow.b_len
        dur_us = max((flow.last - flow.start) * 1e6, 1.0)
        dur_s = dur_us / 1e6

        n_f, n_b = len(flow.f_times), len(flow.b_times)
        b_f, b_b = float(sum(flow.f_len)), float(sum(flow.b_len))

        iat_all = _iats(all_times)
        iat_mean, iat_std, iat_max, _ = _stats(iat_all)
        f_iat_mean, _, _, _ = _stats(_iats(flow.f_times))
        b_iat_mean, _, _, _ = _stats(_iats(flow.b_times))
        pl_mean, pl_std, _, _ = _stats(all_len)
        ttl_mean, ttl_std, _, _ = _stats(flow.ttl_f)

        rows.append({
            "Dst Port": flow.dport,
            "Protocol": flow.proto,
            "Timestamp": pd.to_datetime(flow.start, unit="s"),
            "Flow Duration": dur_us,
            "Tot Fwd Pkts": n_f,
            "Tot Bwd Pkts": n_b,
            "TotLen Fwd Pkts": b_f,
            "TotLen Bwd Pkts": b_b,
            "Flow Byts/s": (b_f + b_b) / dur_s,
            "Flow Pkts/s": (n_f + n_b) / dur_s,
            "Flow IAT Mean": iat_mean,
            "Flow IAT Std": iat_std,
            "Flow IAT Max": iat_max,
            "Fwd IAT Mean": f_iat_mean,
            "Bwd IAT Mean": b_iat_mean,
            "Fwd Pkts/s": n_f / dur_s,
            "Bwd Pkts/s": n_b / dur_s,
            "Pkt Len Mean": pl_mean,
            "Pkt Len Std": pl_std,
            "Pkt Len Var": pl_std ** 2,
            "FIN Flag Cnt": flow.flags["fin"],
            "SYN Flag Cnt": flow.flags["syn"],
            "RST Flag Cnt": flow.flags["rst"],
            "PSH Flag Cnt": flow.flags["psh"],
            "ACK Flag Cnt": flow.flags["ack"],
            "URG Flag Cnt": flow.flags["urg"],
            "Down/Up Ratio": (n_b / n_f) if n_f else 0.0,
            "Pkt Size Avg": pl_mean,
            "Subflow Fwd Pkts": n_f,
            "Subflow Fwd Byts": b_f,
            "Subflow Bwd Pkts": n_b,
            "Subflow Bwd Byts": b_b,
            "Init Fwd Win Byts": flow.init_win_f,
            "Init Bwd Win Byts": flow.init_win_b,
            "Fwd Seg Size Min": flow.seg_min_f if flow.seg_min_f is not None else 0,
            "Active Mean": dur_us,
            "Idle Mean": 0.0,
            # packet-level extras (surfaced in the UI, not consumed by the model)
            "TTL Mean": ttl_mean,
            "TTL Std": ttl_std,
            "IP Frag Count": flow.frag,
            "Retransmissions": flow.retrans,
            "Label": "Benign",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def pcap_summary(df):
    """Short human-readable summary of a parsed capture, for the dashboard."""
    if df.empty:
        return {}
    span = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds()
    return {
        "flows": int(len(df)),
        "duration_sec": round(float(span), 1),
        "unique_dst_ports": int(df["Dst Port"].nunique()),
        "tcp_flows": int((df["Protocol"] == 6).sum()),
        "udp_flows": int((df["Protocol"] == 17).sum()),
        "mean_ttl": round(float(df["TTL Mean"].mean()), 1) if "TTL Mean" in df else 0.0,
        "retransmissions": int(df["Retransmissions"].sum()) if "Retransmissions" in df else 0,
        "fragmented_pkts": int(df["IP Frag Count"].sum()) if "IP Frag Count" in df else 0,
    }

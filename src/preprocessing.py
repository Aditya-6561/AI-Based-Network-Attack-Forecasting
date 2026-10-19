import os
import pandas as pd
import numpy as np
from typing import List

# CSE-CIC-IDS2018 CSVs record timestamps on a 12-hour clock with NO AM/PM marker.
# Captures run roughly 08:00 -> 17:00 local, so any hour < 8 is really an afternoon
# hour and must be shifted by +12. Without this, sort_values() interleaves the
# afternoon before the morning and destroys the temporal ordering the world model
# is supposed to learn from.
AFTERNOON_ROLLOVER_HOUR = 8

# Columns the state builder actually consumes. Restricting the CSV reader to these
# roughly halves ingest time and memory on the 350 MB IDS2018 files.
REQUIRED_COLUMNS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Idle Mean",
    "Label",
]


class TelemetryPreprocessor:
    """
    Data preprocessing pipeline for CSE-CIC-IDS2018 telemetry records.

    Responsibilities:
      - normalise column names
      - repair the 12-hour timestamp ambiguity
      - coerce numerics, neutralise NaN/Inf
      - drop raw IP identifiers so the model cannot memorise host signatures
    """

    def __init__(self, anonymize_ips: bool = True, fix_12h_clock: bool = True):
        self.anonymize_ips = anonymize_ips
        self.fix_12h_clock = fix_12h_clock
        self.n_dropped_off_capture = 0

    # ------------------------------------------------------------------ #
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        # Timestamps that are already datetime64 came from a real clock source (a
        # PCAP's epoch times, for instance), not from an ambiguous 12-hour string.
        # Applying the AM/PM repair to those silently shifts any pre-noon capture
        # forward by 12 hours, so the repair is confined to string input.
        already_datetime = pd.api.types.is_datetime64_any_dtype(df["Timestamp"])

        if already_datetime:
            ts = df["Timestamp"]
        else:
            ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if ts.isna().sum() > len(df) * 0.5:
                ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)

        # A handful of IDS2018 rows carry epoch-zero (1970) timestamps. Left in
        # place they stretch the resample span across decades, turning a 10s
        # windowing into ~10^8 empty buckets and exhausting memory. Keep only
        # rows near the dominant capture date.
        med = ts.median()
        if pd.notna(med):
            off_capture = (ts - med).abs() > pd.Timedelta(days=2)
            self.n_dropped_off_capture = int(off_capture.sum())
            if self.n_dropped_off_capture:
                ts = ts.mask(off_capture)

        if self.fix_12h_clock and not already_datetime:
            hours = ts.dt.hour
            # Only apply when the capture genuinely looks like a 12-hour clock:
            # hours confined to 1..12 and no 13..23 present anywhere.
            valid = hours.dropna()
            if len(valid) and valid.max() <= 12:
                needs_shift = (hours < AFTERNOON_ROLLOVER_HOUR) & hours.notna()
                ts = ts + pd.to_timedelta(np.where(needs_shift, 12, 0), unit="h")

        df["Timestamp"] = ts
        return df

    # ------------------------------------------------------------------ #
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]

        # Some IDS2018 files repeat the CSV header mid-file; those rows carry the
        # literal string "Label" and must not be counted as attacks.
        if "Label" in df.columns:
            df = df[df["Label"].astype(str).str.strip() != "Label"].copy()

        ip_cols = [c for c in df.columns if c.lower() in ("src ip", "dst ip", "flow id")]
        if self.anonymize_ips and ip_cols:
            df = df.drop(columns=ip_cols)

        if "Timestamp" in df.columns:
            df = self._parse_timestamps(df)
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        non_num_cols = ["Timestamp", "Label"]
        num_cols = [c for c in df.columns if c not in non_num_cols]

        # Vectorised numeric coercion (per-column loop was the pipeline bottleneck).
        num_block = df[num_cols].apply(pd.to_numeric, errors="coerce")
        num_block = num_block.replace([np.inf, -np.inf], np.nan)
        num_block = num_block.fillna(num_block.median(numeric_only=True)).fillna(0.0)
        df[num_cols] = num_block

        return df

    # ------------------------------------------------------------------ #
    def load_csv_file(self, fpath: str, nrows: int = None, usecols: List[str] = None) -> pd.DataFrame:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset file not found: {fpath}")
        df = pd.read_csv(fpath, nrows=nrows, low_memory=False, usecols=usecols)
        return self.clean_dataframe(df)

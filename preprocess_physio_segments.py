# -*- coding: utf-8 -*-
r"""
Segment ECG and GSR data into trials using simulator JSON timestamps.

Run examples:
  C:\Users\16638\miniconda3\envs\carla\python.exe preprocess_physio_segments.py --dry-run
  C:\Users\16638\miniconda3\envs\carla\python.exe preprocess_physio_segments.py --subjects T1
  C:\Users\16638\miniconda3\envs\carla\python.exe preprocess_physio_segments.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "\u5b9e\u9a8c\u6570\u636e2"
OUTPUT_DIR = BASE_DIR / "analysis_output_physio_segments"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")

STYLE_ORDER = ["aggressive", "consecutive", "neutral", "self"]


@dataclass(frozen=True)
class TrialEvent:
    subject: str
    condition: str
    scenario: str
    automation_level: str
    style: str
    start_dt: datetime
    end_dt: datetime
    source_json: Path
    cycle_count: int
    cycle_starts: str
    cycle_ends: str

    @property
    def duration_s(self) -> float:
        return (self.end_dt - self.start_dt).total_seconds()

    @property
    def trial_id(self) -> str:
        return f"{self.subject}_{self.condition}_{self.style}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop ECG/GSR data into per-trial segments.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Root folder containing T* subject folders.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--subjects", nargs="*", help="Optional subject IDs, e.g. T1 T2.")
    parser.add_argument("--skip-ecg", action="store_true", help="Do not process TX.txt ECG files.")
    parser.add_argument("--skip-gsr", action="store_true", help="Do not process *_GSR.csv files.")
    parser.add_argument("--dry-run", action="store_true", help="Only build and write/read the event index; do not write signal segments.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing segment files.")
    return parser.parse_args()


def to_local_naive(value: str) -> datetime:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("empty datetime")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(LOCAL_TZ).replace(tzinfo=None)
    return dt


def parse_recording_start(ecg_path: Path) -> tuple[datetime, int]:
    recording_start: datetime | None = None
    data_start_line: int | None = None
    numeric_re = re.compile(r"^\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?(?:\s|$)")

    with ecg_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            if line.startswith("Recording on:"):
                raw = line.split("Recording on:", 1)[1].strip()
                recording_start = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
            if numeric_re.match(line):
                data_start_line = line_no
                break

    if recording_start is None:
        raise ValueError(f"Recording start not found in {ecg_path}")
    if data_start_line is None:
        raise ValueError(f"ECG data start not found in {ecg_path}")
    return recording_start, data_start_line


def classify_event(json_path: Path, payload: dict) -> tuple[str, str, str]:
    parts = [p.lower() for p in json_path.parts]
    style = json_path.parent.name.lower()
    if style not in STYLE_ORDER:
        style = next((s for s in STYLE_ORDER if s in json_path.name.lower()), style)

    if "overtake" in parts:
        return "l4_overtake", "overtake", "L4"

    text = " ".join(
        [
            json_path.name.lower(),
            str(payload.get("session_id", "")).lower(),
            str(payload.get("l3_event_log", "")).lower(),
            str(payload.get("replay_session_log", "")).lower(),
        ]
    )
    if "l3_events" in text or "_l3" in text:
        return "l3_follow", "follow", "L3"
    return "l4_follow", "follow", "L4"


def event_times(payload: dict) -> tuple[datetime, datetime, list[datetime], list[datetime]]:
    starts: list[datetime] = []
    ends: list[datetime] = []

    cycles = payload.get("cycles")
    if isinstance(cycles, list):
        for cycle in cycles:
            if not isinstance(cycle, dict):
                continue
            start_raw = cycle.get("experiment_start_system_local")
            end_raw = cycle.get("experiment_end_system_local")
            if start_raw and end_raw:
                starts.append(to_local_naive(start_raw))
                ends.append(to_local_naive(end_raw))

    if not starts or not ends:
        start_raw = payload.get("experiment_start_system_local")
        end_raw = payload.get("experiment_end_system_local")
        if start_raw and end_raw:
            starts = [to_local_naive(start_raw)]
            ends = [to_local_naive(end_raw)]

    if not starts or not ends:
        raise ValueError("No experiment start/end timestamps found")

    return min(starts), max(ends), starts, ends


def discover_events(subject_dir: Path) -> list[TrialEvent]:
    events: list[TrialEvent] = []
    for json_path in sorted(subject_dir.glob("*/*/*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            start_dt, end_dt, starts, ends = event_times(payload)
            condition, scenario, automation_level = classify_event(json_path, payload)
        except Exception as exc:
            print(f"[warn] skip JSON {json_path}: {exc}", file=sys.stderr)
            continue

        events.append(
            TrialEvent(
                subject=subject_dir.name,
                condition=condition,
                scenario=scenario,
                automation_level=automation_level,
                style=json_path.parent.name.lower(),
                start_dt=start_dt,
                end_dt=end_dt,
                source_json=json_path,
                cycle_count=len(starts),
                cycle_starts=";".join(dt.isoformat(sep=" ") for dt in starts),
                cycle_ends=";".join(dt.isoformat(sep=" ") for dt in ends),
            )
        )

    events.sort(key=lambda ev: ev.start_dt)
    return events


def trial_index_rows(events: list[TrialEvent]) -> list[dict]:
    return [
        {
            "subject": ev.subject,
            "trial_id": ev.trial_id,
            "condition": ev.condition,
            "scenario": ev.scenario,
            "automation_level": ev.automation_level,
            "style": ev.style,
            "start_time": ev.start_dt.isoformat(sep=" "),
            "end_time": ev.end_dt.isoformat(sep=" "),
            "duration_s": round(ev.duration_s, 6),
            "cycle_count": ev.cycle_count,
            "cycle_starts": ev.cycle_starts,
            "cycle_ends": ev.cycle_ends,
            "source_json": str(ev.source_json),
        }
        for ev in events
    ]


def load_ecg(ecg_path: Path) -> tuple[pd.DataFrame, datetime]:
    recording_start, data_start_line = parse_recording_start(ecg_path)
    df = pd.read_csv(
        ecg_path,
        sep=r"\s+",
        skiprows=data_start_line,
        header=None,
        usecols=[0, 1],
        names=["ecg_relative_s", "ECG_mV"],
        engine="c",
        low_memory=False,
    )
    df["ecg_relative_s"] = pd.to_numeric(df["ecg_relative_s"], errors="coerce")
    df["ECG_mV"] = pd.to_numeric(df["ECG_mV"], errors="coerce")
    df = df.dropna(subset=["ecg_relative_s", "ECG_mV"])
    if df.empty:
        raise ValueError(f"No numeric ECG samples found in {ecg_path}")
    return df, recording_start


def crop_ecg(ecg_df: pd.DataFrame, recording_start: datetime, event: TrialEvent) -> pd.DataFrame:
    start_rel = (event.start_dt - recording_start).total_seconds()
    end_rel = (event.end_dt - recording_start).total_seconds()
    mask = (ecg_df["ecg_relative_s"] >= start_rel) & (ecg_df["ecg_relative_s"] <= end_rel)
    segment = ecg_df.loc[mask].copy()
    segment["elapsed_s"] = segment["ecg_relative_s"] - start_rel
    timestamps = recording_start + pd.to_timedelta(segment["ecg_relative_s"], unit="s")
    segment.insert(0, "timestamp", timestamps.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
    segment = segment[["timestamp", "elapsed_s", "ecg_relative_s", "ECG_mV"]]
    return segment


def infer_gsr_timestamps(raw: pd.DataFrame, time_col: str) -> pd.Series:
    base_time = pd.to_datetime(raw[time_col], errors="coerce", format="mixed")
    if base_time.isna().any():
        bad = int(base_time.isna().sum())
        raise ValueError(f"{bad} invalid GSR timestamps")

    unique_times = pd.Series(base_time.drop_duplicates().sort_values().to_numpy())
    positive_diffs = unique_times.diff().dt.total_seconds().dropna()
    group_span_s = 60.0
    if not positive_diffs.empty and positive_diffs.median() <= 1.5:
        group_span_s = 1.0

    counts = base_time.value_counts(sort=False)
    full_count = int(counts.max())
    if full_count <= 0:
        raise ValueError("Cannot infer GSR sampling rate")
    fs = full_count / group_span_s

    ordinal = raw.groupby(base_time, sort=False).cumcount().astype(float)
    first_time = base_time.iloc[0]
    start_offsets = pd.Series(0.0, index=raw.index)

    first_count = counts.loc[first_time]
    if first_count < full_count:
        start_offsets.loc[base_time == first_time] = group_span_s - first_count / fs

    seconds = start_offsets + ordinal / fs
    return base_time + pd.to_timedelta(seconds, unit="s")


def load_gsr(gsr_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(gsr_path)
    time_cols = [c for c in raw.columns if "time" in c.lower()]
    value_cols = [c for c in raw.columns if c not in time_cols]
    if not time_cols or not value_cols:
        raise ValueError(f"Cannot identify GSR/time columns in {gsr_path}")

    time_col = time_cols[0]
    value_col = value_cols[0]
    df = pd.DataFrame({"GSR": pd.to_numeric(raw[value_col], errors="coerce")})
    df["timestamp_dt"] = infer_gsr_timestamps(raw, time_col)
    df = df.dropna(subset=["GSR", "timestamp_dt"])
    return df


def load_gsr_files(gsr_paths: list[Path]) -> pd.DataFrame:
    frames = [load_gsr(path) for path in gsr_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp_dt").drop_duplicates(subset=["timestamp_dt"], keep="first")
    return df.reset_index(drop=True)


def crop_gsr(gsr_df: pd.DataFrame, event: TrialEvent) -> pd.DataFrame:
    mask = (gsr_df["timestamp_dt"] >= event.start_dt) & (gsr_df["timestamp_dt"] <= event.end_dt)
    segment = gsr_df.loc[mask].copy()
    segment["elapsed_s"] = (segment["timestamp_dt"] - event.start_dt).dt.total_seconds()
    segment.insert(0, "timestamp", segment["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
    segment = segment[["timestamp", "elapsed_s", "GSR"]]
    return segment


def write_segment(segment: pd.DataFrame, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segment.to_csv(out_path, index=False, encoding="utf-8-sig")
    return True


def find_subject_dirs(data_root: Path, selected: list[str] | None) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and re.fullmatch(r"T\d+", p.name)], key=lambda p: int(p.name[1:]))
    if selected:
        wanted = set(selected)
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def process_subject(subject_dir: Path, out_dir: Path, args: argparse.Namespace) -> list[dict]:
    events = discover_events(subject_dir)
    rows = trial_index_rows(events)
    if not events:
        print(f"[warn] no trial events found for {subject_dir.name}", file=sys.stderr)
        return rows

    subject_out = out_dir / subject_dir.name

    ecg_df: pd.DataFrame | None = None
    ecg_start: datetime | None = None
    if not args.skip_ecg and not args.dry_run:
        ecg_path = subject_dir / f"{subject_dir.name}.txt"
        if ecg_path.exists():
            print(f"[info] loading ECG {ecg_path}")
            try:
                ecg_df, ecg_start = load_ecg(ecg_path)
            except Exception as exc:
                print(f"[warn] cannot load ECG {ecg_path}: {exc}", file=sys.stderr)
        else:
            print(f"[warn] ECG file not found: {ecg_path}", file=sys.stderr)

    gsr_df: pd.DataFrame | None = None
    if not args.skip_gsr and not args.dry_run:
        gsr_files = sorted(subject_dir.glob("*_GSR.csv"))
        if gsr_files:
            print(f"[info] loading GSR files={len(gsr_files)}")
            try:
                gsr_df = load_gsr_files(gsr_files)
            except Exception as exc:
                print(f"[warn] cannot load GSR files in {subject_dir}: {exc}", file=sys.stderr)
        else:
            print(f"[warn] GSR file not found in {subject_dir}", file=sys.stderr)

    for row, event in zip(rows, events):
        row["ecg_rows"] = 0
        row["ecg_segment_file"] = ""
        row["ecg_written"] = False
        row["gsr_rows"] = 0
        row["gsr_segment_file"] = ""
        row["gsr_written"] = False

        if ecg_df is not None and ecg_start is not None:
            segment = crop_ecg(ecg_df, ecg_start, event)
            ecg_out = subject_out / "ecg" / f"{event.trial_id}_ecg.csv"
            written = write_segment(segment, ecg_out, args.overwrite)
            row["ecg_rows"] = len(segment)
            row["ecg_segment_file"] = str(ecg_out)
            row["ecg_written"] = written
        if gsr_df is not None:
            segment = crop_gsr(gsr_df, event)
            gsr_out = subject_out / "gsr" / f"{event.trial_id}_gsr.csv"
            written = write_segment(segment, gsr_out, args.overwrite)
            row["gsr_rows"] = len(segment)
            row["gsr_segment_file"] = str(gsr_out)
            row["gsr_written"] = written

    return rows


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = find_subject_dirs(args.data_root, args.subjects)
    if not subject_dirs:
        print("No subject folders found.", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for subject_dir in subject_dirs:
        print(f"[info] subject {subject_dir.name}")
        rows = process_subject(subject_dir, out_dir, args)
        all_rows.extend(rows)

    index = pd.DataFrame(all_rows)
    index_path = out_dir / ("trial_time_index_dry_run.csv" if args.dry_run else "trial_time_index.csv")
    index.to_csv(index_path, index=False, encoding="utf-8-sig")

    print(f"[done] subjects={len(subject_dirs)} trials={len(index)}")
    print(f"[done] index={index_path}")
    if args.dry_run:
        print("[done] dry run only; signal segment files were not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

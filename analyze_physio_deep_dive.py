# -*- coding: utf-8 -*-
r"""
Deep-dive analyses for Section 4.5 physiological stress response.

This script builds on analyze_physio_stress_response.py outputs and adds:
  1) within-subject z-score normalization,
  2) a composite physiological load index,
  3) L4 overtaking cycle/window EDA analyses,
  4) L3 space-press-centered EDA analyses,
  5) HRV quality-control flags.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_physio_deep_dive.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from analyze_physio_stress_response import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    INDEX_FILE,
    OUTPUT_DIR as BASIC_OUTPUT_DIR,
    STYLE_COLORS,
    STYLE_LABELS,
    STYLE_ORDER,
    eda_features_from_arrays,
    load_gsr_segment,
    p_text,
    parse_dt,
    parse_dt_list,
    setup_font,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis_output_physio_deep_dive"
BASIC_FEATURE_FILE = BASIC_OUTPUT_DIR / "4.5_physio_trial_features_combined.csv"
BASIC_CYCLE_FILE = BASIC_OUTPUT_DIR / "4.5.1_EDA_cycle_features.csv"

EDA_METRICS = ["eda_peak_amp_max", "eda_peak_rate_per_min", "eda_phasic_auc_per_min"]
HRV_METRICS = ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50"]
TRIAL_Z_METRICS = EDA_METRICS + HRV_METRICS
LOAD_COMPONENTS = ["eda_peak_amp_max_z", "eda_phasic_auc_per_min_z", "mean_hr_bpm_z", "rmssd_load_z"]
WINDOW_ORDER = ["start_0_10s", "middle_10s", "end_last_10s"]
WINDOW_LABELS = {
    "start_0_10s": "Cycle start 0-10 s",
    "middle_10s": "Cycle middle 10 s",
    "end_last_10s": "Cycle end last 10 s",
}
PRESS_WINDOW_ORDER = ["pre_5s", "post_0_10s", "post_10_20s"]
PRESS_WINDOW_LABELS = {
    "pre_5s": "Pre-press 5 s",
    "post_0_10s": "Post-press 0-10 s",
    "post_10_20s": "Post-press 10-20 s",
}


def style_label(style: str) -> str:
    return STYLE_LABELS.get(style, str(style).title())


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, str(condition))


def safe_subject_zscore(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        values = pd.to_numeric(out[metric], errors="coerce")
        mean = values.groupby(out["subject"]).transform("mean")
        std = values.groupby(out["subject"]).transform(lambda s: s.std(ddof=0))
        z = (values - mean) / std.replace(0, np.nan)
        out[f"{metric}_z"] = z.fillna(0.0)
    return out


def add_hrv_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hrv_qc_pass"] = (
        out["mean_hr_bpm"].between(45, 130)
        & out["rmssd_ms"].between(5, 200)
        & (out["valid_rr_count"] >= 30)
    )
    out["hrv_qc_reason"] = ""
    out.loc[~out["mean_hr_bpm"].between(45, 130), "hrv_qc_reason"] += "heart_rate_out_of_range;"
    out.loc[~out["rmssd_ms"].between(5, 200), "hrv_qc_reason"] += "rmssd_out_of_range;"
    out.loc[out["valid_rr_count"] < 30, "hrv_qc_reason"] += "few_rr_intervals;"
    return out


def add_load_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rmssd_load_z"] = -out["rmssd_ms_z"]
    out["sdnn_load_z"] = -out["sdnn_ms_z"]
    out["eda_load_z"] = out[["eda_peak_amp_max_z", "eda_phasic_auc_per_min_z"]].mean(axis=1)
    out["hrv_load_z"] = out[["mean_hr_bpm_z", "rmssd_load_z"]].mean(axis=1)
    out["physio_load_z"] = out[LOAD_COMPONENTS].mean(axis=1)
    out.loc[~out["hrv_qc_pass"], ["mean_hr_bpm_z", "rmssd_load_z", "sdnn_load_z", "hrv_load_z", "physio_load_z"]] = np.nan
    out["physio_load_z"] = out[LOAD_COMPONENTS].mean(axis=1, skipna=True)
    return out


def mean_se_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_subjects"] = int(sub["subject"].nunique()) if "subject" in sub else len(sub)
        row["n_rows"] = int(len(sub))
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_condition_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    comparisons = [
        ("L3 vs L4 Following", "l3_follow", "l4_follow"),
        ("L4 Overtaking vs L4 Following", "l4_overtake", "l4_follow"),
    ]
    rows = []
    for metric in metrics:
        grouped = df.groupby(["subject", "condition"], as_index=False)[metric].mean()
        pivot = grouped.pivot(index="subject", columns="condition", values=metric)
        for label, a, b in comparisons:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            paired = pivot[[a, b]].dropna()
            if len(paired) < 2:
                continue
            diff = paired[a] - paired[b]
            if diff.std(ddof=1) > 0:
                t_stat, p_value = stats.ttest_rel(paired[a], paired[b])
                dz = float(diff.mean() / diff.std(ddof=1))
            else:
                t_stat, p_value, dz = np.nan, np.nan, np.nan
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            rows.append(
                {
                    "metric": metric,
                    "comparison": label,
                    "n": int(len(paired)),
                    "mean_a": float(paired[a].mean()),
                    "mean_b": float(paired[b].mean()),
                    "mean_diff_a_minus_b": float(diff.mean()),
                    "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p_ttest": float(p_value) if pd.notna(p_value) else np.nan,
                    "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
                    "p_wilcoxon": float(w_p) if pd.notna(w_p) else np.nan,
                    "cohens_dz": dz,
                }
            )
    return pd.DataFrame(rows)


def friedman_style_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        grouped = df.groupby(["subject", "condition", "style"], as_index=False)[metric].mean()
        for condition, sub in grouped.groupby("condition", sort=False):
            pivot = sub.pivot(index="subject", columns="style", values=metric)
            if not all(style in pivot.columns for style in STYLE_ORDER):
                continue
            pivot = pivot[STYLE_ORDER].dropna()
            if len(pivot) < 3:
                continue
            try:
                stat, p_value = stats.friedmanchisquare(*(pivot[style] for style in STYLE_ORDER))
            except ValueError:
                stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "n": int(len(pivot)),
                    "friedman_chi2": float(stat) if pd.notna(stat) else np.nan,
                    "p_friedman": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def self_vs_other_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        grouped = df.groupby(["subject", "condition", "style"], as_index=False)[metric].mean()
        self_df = grouped[grouped["style"] == "self"].rename(columns={metric: "self_value"})
        other_df = (
            grouped[grouped["style"] != "self"]
            .groupby(["subject", "condition"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "other_mean"})
        )
        merged = self_df.merge(other_df, on=["subject", "condition"], how="inner")
        merged = merged.dropna(subset=["self_value", "other_mean"])
        for condition, sub in merged.groupby("condition", sort=False):
            diff = sub["self_value"] - sub["other_mean"]
            if len(sub) > 1 and diff.std(ddof=1) > 0:
                t_stat, p_value = stats.ttest_rel(sub["self_value"], sub["other_mean"])
                dz = float(diff.mean() / diff.std(ddof=1))
            else:
                t_stat, p_value, dz = np.nan, np.nan, np.nan
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "n": int(len(sub)),
                    "self_mean": float(sub["self_value"].mean()),
                    "other_mean": float(sub["other_mean"].mean()),
                    "self_minus_other": float(diff.mean()),
                    "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p_ttest": float(p_value) if pd.notna(p_value) else np.nan,
                    "cohens_dz": dz,
                }
            )
    return pd.DataFrame(rows)


def plot_condition_style(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    setup_font()
    condition_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    fig, ax = plt.subplots(figsize=(8.7, 5.0), dpi=180)
    x = np.arange(len(CONDITION_ORDER))
    for style in STYLE_ORDER:
        sub = summary[summary["style"] == style].copy()
        sub["_rank"] = sub["condition"].map(condition_rank)
        sub = sub.sort_values("_rank").set_index("condition")
        means = [sub.loc[c, f"{metric}_mean"] if c in sub.index else np.nan for c in CONDITION_ORDER]
        ses = [sub.loc[c, f"{metric}_se"] if c in sub.index else np.nan for c in CONDITION_ORDER]
        ax.errorbar(
            x,
            means,
            yerr=ses,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=STYLE_COLORS.get(style),
            label=style_label(style),
        )
    ax.axhline(0, color="#888888", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in CONDITION_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by condition and style")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cycle(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    setup_font()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    x = np.array([1, 2, 3])
    for style in STYLE_ORDER:
        sub = summary[summary["style"] == style].sort_values("cycle_index").set_index("cycle_index")
        means = [sub.loc[i, f"{metric}_mean"] if i in sub.index else np.nan for i in x]
        ses = [sub.loc[i, f"{metric}_se"] if i in sub.index else np.nan for i in x]
        ax.errorbar(
            x,
            means,
            yerr=ses,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=STYLE_COLORS.get(style),
            label=style_label(style),
        )
    ax.axhline(0, color="#888888", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xlabel("Overtaking cycle")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across L4 overtaking cycles")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_windows(summary: pd.DataFrame, window_order: list[str], labels: dict[str, str], metric: str, ylabel: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    setup_font()
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    x = np.arange(len(window_order))
    for style in STYLE_ORDER:
        sub = summary[summary["style"] == style].set_index("window")
        means = [sub.loc[w, f"{metric}_mean"] if w in sub.index else np.nan for w in window_order]
        ses = [sub.loc[w, f"{metric}_se"] if w in sub.index else np.nan for w in window_order]
        ax.errorbar(
            x,
            means,
            yerr=ses,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=STYLE_COLORS.get(style),
            label=style_label(style),
        )
    ax.axhline(0, color="#888888", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[w] for w in window_order], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by event window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "condition" in out.columns:
        out["condition_label"] = out["condition"].map(condition_label)
    if "style" in out.columns:
        out["style_label"] = out["style"].map(style_label)
    return out


def prepare_trial_deep_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(BASIC_FEATURE_FILE)
    df = add_hrv_qc(df)
    df = safe_subject_zscore(df, TRIAL_Z_METRICS)
    df = add_load_indices(df)
    df = add_label_columns(df)

    metrics = [
        "eda_load_z",
        "hrv_load_z",
        "physio_load_z",
        "eda_peak_amp_max_z",
        "eda_phasic_auc_per_min_z",
        "mean_hr_bpm_z",
        "rmssd_load_z",
    ]
    summary = mean_se_summary(df, ["condition", "style"], metrics)
    summary = add_label_columns(summary)
    condition_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    style_rank = {s: i for i, s in enumerate(STYLE_ORDER)}
    summary["_condition_rank"] = summary["condition"].map(condition_rank)
    summary["_style_rank"] = summary["style"].map(style_rank)
    summary = summary.sort_values(["_condition_rank", "_style_rank"]).drop(columns=["_condition_rank", "_style_rank"])

    paired = paired_condition_tests(df, metrics)
    friedman = friedman_style_tests(df, ["physio_load_z", "eda_load_z", "hrv_load_z"])
    self_tests = self_vs_other_tests(df, ["physio_load_z", "eda_load_z", "hrv_load_z"])
    return df, summary, paired, friedman, self_tests


def prepare_cycle_deep_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycle = pd.read_csv(BASIC_CYCLE_FILE)
    cycle = cycle[cycle["condition"] == "l4_overtake"].copy()
    cycle = safe_subject_zscore(cycle, EDA_METRICS)
    cycle["cycle_eda_load_z"] = cycle[["eda_peak_amp_max_z", "eda_phasic_auc_per_min_z", "eda_peak_rate_per_min_z"]].mean(axis=1)
    cycle = add_label_columns(cycle)

    summary = mean_se_summary(cycle, ["cycle_index", "style"], ["cycle_eda_load_z", "eda_peak_amp_max_z", "eda_phasic_auc_per_min_z"])
    summary = add_label_columns(summary)
    summary["_style_rank"] = summary["style"].map({s: i for i, s in enumerate(STYLE_ORDER)})
    summary = summary.sort_values(["cycle_index", "_style_rank"]).drop(columns=["_style_rank"])

    rows = []
    metric = "cycle_eda_load_z"
    subject_cycle = cycle.groupby(["subject", "cycle_index"], as_index=False)[metric].mean()
    pivot_cycle = subject_cycle.pivot(index="subject", columns="cycle_index", values=metric)
    if all(c in pivot_cycle.columns for c in [1, 2, 3]):
        pivot_cycle = pivot_cycle[[1, 2, 3]].dropna()
        stat, p_value = stats.friedmanchisquare(pivot_cycle[1], pivot_cycle[2], pivot_cycle[3])
        rows.append({"analysis": "cycle_main_effect", "metric": metric, "n": len(pivot_cycle), "stat": stat, "p_value": p_value})
        for a, b in [(2, 1), (3, 1), (3, 2)]:
            diff = pivot_cycle[a] - pivot_cycle[b]
            t_stat, t_p = stats.ttest_rel(pivot_cycle[a], pivot_cycle[b])
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            rows.append(
                {
                    "analysis": f"cycle_{a}_minus_{b}",
                    "metric": metric,
                    "n": len(pivot_cycle),
                    "mean_diff": diff.mean(),
                    "stat": t_stat,
                    "p_value": t_p,
                    "wilcoxon_stat": w_stat,
                    "p_wilcoxon": w_p,
                }
            )

    grouped = cycle.groupby(["subject", "cycle_index", "style"], as_index=False)[metric].mean()
    for cycle_index, sub in grouped.groupby("cycle_index"):
        pivot = sub.pivot(index="subject", columns="style", values=metric)
        if all(style in pivot.columns for style in STYLE_ORDER):
            pivot = pivot[STYLE_ORDER].dropna()
            stat, p_value = stats.friedmanchisquare(*(pivot[style] for style in STYLE_ORDER))
            rows.append(
                {
                    "analysis": f"style_effect_cycle_{cycle_index}",
                    "metric": metric,
                    "n": len(pivot),
                    "stat": stat,
                    "p_value": p_value,
                }
            )
    return cycle, summary, pd.DataFrame(rows)


def gsr_window_features(gsr: pd.DataFrame, start_s: float, end_s: float) -> dict:
    if end_s <= start_s:
        return eda_features_from_arrays(np.array([]), np.array([]))
    mask = (gsr["elapsed_s"] >= start_s) & (gsr["elapsed_s"] <= end_s)
    return eda_features_from_arrays(gsr.loc[mask, "elapsed_s"].to_numpy(float), gsr.loc[mask, "GSR"].to_numpy(float))


def prepare_overtake_window_features(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    overtake = index[index["condition"] == "l4_overtake"].copy()
    for _, row in overtake.iterrows():
        gsr_path = Path(str(row["gsr_segment_file"]))
        if not gsr_path.exists():
            continue
        gsr = load_gsr_segment(gsr_path)
        trial_start = parse_dt(row["start_time"])
        cycle_starts = parse_dt_list(row["cycle_starts"])
        cycle_ends = parse_dt_list(row["cycle_ends"])
        for cycle_index, (cycle_start, cycle_end) in enumerate(zip(cycle_starts, cycle_ends), start=1):
            cycle_start_s = (cycle_start - trial_start).total_seconds()
            cycle_end_s = (cycle_end - trial_start).total_seconds()
            mid_s = (cycle_start_s + cycle_end_s) / 2.0
            windows = {
                "start_0_10s": (cycle_start_s, min(cycle_start_s + 10.0, cycle_end_s)),
                "middle_10s": (max(cycle_start_s, mid_s - 5.0), min(cycle_end_s, mid_s + 5.0)),
                "end_last_10s": (max(cycle_start_s, cycle_end_s - 10.0), cycle_end_s),
            }
            for window, (start_s, end_s) in windows.items():
                features = gsr_window_features(gsr, start_s, end_s)
                rows.append(
                    {
                        "subject": row["subject"],
                        "trial_id": row["trial_id"],
                        "condition": row["condition"],
                        "style": row["style"],
                        "cycle_index": cycle_index,
                        "window": window,
                        "window_label": WINDOW_LABELS[window],
                        "window_start_s": start_s,
                        "window_end_s": end_s,
                        **features,
                    }
                )
    window_df = pd.DataFrame(rows)
    if window_df.empty:
        return window_df, pd.DataFrame(), pd.DataFrame()
    window_df = safe_subject_zscore(window_df, EDA_METRICS)
    window_df["window_eda_load_z"] = window_df[["eda_peak_amp_max_z", "eda_phasic_auc_per_min_z", "eda_peak_rate_per_min_z"]].mean(axis=1)
    window_df = add_label_columns(window_df)

    subject_style_window = window_df.groupby(["subject", "style", "window"], as_index=False)["window_eda_load_z"].mean()
    summary = mean_se_summary(subject_style_window, ["window", "style"], ["window_eda_load_z"])
    summary = add_label_columns(summary)
    summary["_window_rank"] = summary["window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    summary["_style_rank"] = summary["style"].map({s: i for i, s in enumerate(STYLE_ORDER)})
    summary = summary.sort_values(["_window_rank", "_style_rank"]).drop(columns=["_window_rank", "_style_rank"])

    rows = []
    subject_window = window_df.groupby(["subject", "window"], as_index=False)["window_eda_load_z"].mean()
    pivot = subject_window.pivot(index="subject", columns="window", values="window_eda_load_z")
    if all(window in pivot.columns for window in WINDOW_ORDER):
        pivot = pivot[WINDOW_ORDER].dropna()
        stat, p_value = stats.friedmanchisquare(*(pivot[w] for w in WINDOW_ORDER))
        rows.append({"analysis": "window_main_effect", "metric": "window_eda_load_z", "n": len(pivot), "stat": stat, "p_value": p_value})
        for a, b in [("middle_10s", "start_0_10s"), ("end_last_10s", "start_0_10s"), ("end_last_10s", "middle_10s")]:
            diff = pivot[a] - pivot[b]
            t_stat, t_p = stats.ttest_rel(pivot[a], pivot[b])
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            rows.append(
                {
                    "analysis": f"{a}_minus_{b}",
                    "metric": "window_eda_load_z",
                    "n": len(pivot),
                    "mean_diff": diff.mean(),
                    "stat": t_stat,
                    "p_value": t_p,
                    "wilcoxon_stat": w_stat,
                    "p_wilcoxon": w_p,
                }
            )
    return window_df, summary, pd.DataFrame(rows)


def read_space_presses(source_json: Path) -> list[float]:
    try:
        payload = json.loads(source_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    seconds: list[float] = []
    for press in payload.get("space_presses", []) or []:
        if not isinstance(press, dict):
            continue
        value = press.get("seconds_since_experiment_start")
        try:
            seconds.append(float(value))
        except (TypeError, ValueError):
            continue
    return seconds


def prepare_l3_space_press_features(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    l3 = index[index["condition"] == "l3_follow"].copy()
    for _, row in l3.iterrows():
        press_times = read_space_presses(Path(str(row["source_json"])))
        if not press_times:
            continue
        gsr_path = Path(str(row["gsr_segment_file"]))
        if not gsr_path.exists():
            continue
        gsr = load_gsr_segment(gsr_path)
        duration = float(row["duration_s"])
        for press_index, press_s in enumerate(press_times, start=1):
            windows = {
                "pre_5s": (max(0.0, press_s - 5.0), press_s),
                "post_0_10s": (press_s, min(duration, press_s + 10.0)),
                "post_10_20s": (min(duration, press_s + 10.0), min(duration, press_s + 20.0)),
            }
            for window, (start_s, end_s) in windows.items():
                features = gsr_window_features(gsr, start_s, end_s)
                rows.append(
                    {
                        "subject": row["subject"],
                        "trial_id": row["trial_id"],
                        "condition": row["condition"],
                        "style": row["style"],
                        "press_index": press_index,
                        "press_s": press_s,
                        "window": window,
                        "window_label": PRESS_WINDOW_LABELS[window],
                        "window_start_s": start_s,
                        "window_end_s": end_s,
                        **features,
                    }
                )
    press_df = pd.DataFrame(rows)
    if press_df.empty:
        return press_df, pd.DataFrame(), pd.DataFrame()
    press_df = safe_subject_zscore(press_df, EDA_METRICS)
    press_df["press_eda_load_z"] = press_df[["eda_peak_amp_max_z", "eda_phasic_auc_per_min_z", "eda_peak_rate_per_min_z"]].mean(axis=1)
    press_df = add_label_columns(press_df)

    subject_style_window = press_df.groupby(["subject", "style", "window"], as_index=False)["press_eda_load_z"].mean()
    summary = mean_se_summary(subject_style_window, ["window", "style"], ["press_eda_load_z"])
    summary = add_label_columns(summary)
    summary["_window_rank"] = summary["window"].map({w: i for i, w in enumerate(PRESS_WINDOW_ORDER)})
    summary["_style_rank"] = summary["style"].map({s: i for i, s in enumerate(STYLE_ORDER)})
    summary = summary.sort_values(["_window_rank", "_style_rank"]).drop(columns=["_window_rank", "_style_rank"])

    rows = []
    subject_window = press_df.groupby(["subject", "window"], as_index=False)["press_eda_load_z"].mean()
    pivot = subject_window.pivot(index="subject", columns="window", values="press_eda_load_z")
    if all(window in pivot.columns for window in PRESS_WINDOW_ORDER):
        pivot = pivot[PRESS_WINDOW_ORDER].dropna()
        stat, p_value = stats.friedmanchisquare(*(pivot[w] for w in PRESS_WINDOW_ORDER))
        rows.append({"analysis": "space_press_window_main_effect", "metric": "press_eda_load_z", "n": len(pivot), "stat": stat, "p_value": p_value})
        for a, b in [("post_0_10s", "pre_5s"), ("post_10_20s", "pre_5s"), ("post_10_20s", "post_0_10s")]:
            diff = pivot[a] - pivot[b]
            t_stat, t_p = stats.ttest_rel(pivot[a], pivot[b])
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            rows.append(
                {
                    "analysis": f"{a}_minus_{b}",
                    "metric": "press_eda_load_z",
                    "n": len(pivot),
                    "mean_diff": diff.mean(),
                    "stat": t_stat,
                    "p_value": t_p,
                    "wilcoxon_stat": w_stat,
                    "p_wilcoxon": w_p,
                }
            )
    return press_df, summary, pd.DataFrame(rows)


def write_deep_summary(
    trial_df: pd.DataFrame,
    paired: pd.DataFrame,
    friedman: pd.DataFrame,
    self_tests: pd.DataFrame,
    cycle_tests: pd.DataFrame,
    window_tests: pd.DataFrame,
    press_df: pd.DataFrame,
    press_tests: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("4.5 生理数据深挖分析结果摘要")
    lines.append("")
    lines.append("一、分析处理")
    lines.append("- 对 EDA 与 HRV 指标进行被试内 z-score 标准化，以削弱个体基础皮电和心率差异。")
    lines.append("- 构建综合生理负荷指数：PhysioLoad = mean[z(EDA peak), z(EDA phasic AUC), z(HR), -z(RMSSD)]。数值越高表示生理负荷越高。")
    lines.append("- HRV 质控标准：45 <= mean HR <= 130 bpm，5 <= RMSSD <= 200 ms，valid RR >= 30。")
    lines.append(f"- HRV 质控通过 trial：{int(trial_df['hrv_qc_pass'].sum())}/{len(trial_df)}。")
    lines.append("")

    lines.append("二、综合生理负荷的条件差异")
    for metric in ["physio_load_z", "eda_load_z", "hrv_load_z"]:
        sub = paired[paired["metric"] == metric]
        if sub.empty:
            continue
        lines.append(f"- {metric}:")
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['comparison']}，差值={row['mean_diff_a_minus_b']:.3f}，"
                f"t检验 p={p_text(row['p_ttest'])}，Wilcoxon p={p_text(row['p_wilcoxon'])}。"
            )
    lines.append("")

    lines.append("三、风格差异与 Self 效应")
    sub = friedman[friedman["metric"] == "physio_load_z"]
    for _, row in sub.iterrows():
        lines.append(
            f"- {row['condition_label']} 风格主效应：Friedman chi2={row['friedman_chi2']:.3f}，p={p_text(row['p_friedman'])}。"
        )
    sub = self_tests[self_tests["metric"] == "physio_load_z"]
    for _, row in sub.iterrows():
        lines.append(
            f"- {row['condition_label']} Self vs Others：差值={row['self_minus_other']:.3f}，p={p_text(row['p_ttest'])}。"
        )
    lines.append("")

    lines.append("四、L4 超车 cycle/window 深挖")
    for _, row in cycle_tests.iterrows():
        if row["analysis"] == "cycle_main_effect":
            lines.append(f"- 超车 cycle 主效应：Friedman/统计量={row['stat']:.3f}，p={p_text(row['p_value'])}。")
        elif str(row["analysis"]).startswith("cycle_"):
            lines.append(
                f"- {row['analysis']}：差值={row.get('mean_diff', np.nan):.3f}，"
                f"t检验 p={p_text(row['p_value'])}，Wilcoxon p={p_text(row.get('p_wilcoxon', np.nan))}。"
            )
    for _, row in window_tests.iterrows():
        if row["analysis"] == "window_main_effect":
            lines.append(f"- 超车窗口主效应：Friedman/统计量={row['stat']:.3f}，p={p_text(row['p_value'])}。")
        elif "minus" in str(row["analysis"]):
            lines.append(
                f"- {row['analysis']}：差值={row.get('mean_diff', np.nan):.3f}，"
                f"t检验 p={p_text(row['p_value'])}，Wilcoxon p={p_text(row.get('p_wilcoxon', np.nan))}。"
            )
    lines.append("")

    lines.append("五、L3 按键窗口分析")
    lines.append(f"- 检测到包含 space press 的窗口记录：{len(press_df)} 行。")
    for _, row in press_tests.iterrows():
        if row["analysis"] == "space_press_window_main_effect":
            lines.append(f"- 按键窗口主效应：Friedman/统计量={row['stat']:.3f}，p={p_text(row['p_value'])}。")
        elif "minus" in str(row["analysis"]):
            lines.append(
                f"- {row['analysis']}：差值={row.get('mean_diff', np.nan):.3f}，"
                f"t检验 p={p_text(row['p_value'])}，Wilcoxon p={p_text(row.get('p_wilcoxon', np.nan))}。"
            )
    lines.append("")
    lines.append("写作建议：优先报告有稳定统计支持的条件/任务差异；若风格或 Self 效应不显著，应写成“未观察到稳定生理差异”。")
    (OUTPUT_DIR / "4.5_deep_result_summary.txt").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    if not BASIC_FEATURE_FILE.exists():
        print(f"Missing basic feature file: {BASIC_FEATURE_FILE}", file=sys.stderr)
        return 1
    if not INDEX_FILE.exists():
        print(f"Missing trial index: {INDEX_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] trial-level z-score and PhysioLoad")
    trial_df, trial_summary, paired, friedman, self_tests = prepare_trial_deep_features()
    trial_df.to_csv(OUTPUT_DIR / "4.5_deep_trial_zscored_physio_load.csv", index=False, encoding="utf-8-sig")
    trial_summary.to_csv(OUTPUT_DIR / "4.5_deep_trial_condition_style_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUTPUT_DIR / "4.5_deep_trial_condition_paired_tests.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(OUTPUT_DIR / "4.5_deep_trial_style_friedman_tests.csv", index=False, encoding="utf-8-sig")
    self_tests.to_csv(OUTPUT_DIR / "4.5_deep_trial_self_vs_other_tests.csv", index=False, encoding="utf-8-sig")

    print("[info] L4 overtaking cycle analysis")
    cycle_df, cycle_summary, cycle_tests = prepare_cycle_deep_features()
    cycle_df.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_cycle_zscored.csv", index=False, encoding="utf-8-sig")
    cycle_summary.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_cycle_summary.csv", index=False, encoding="utf-8-sig")
    cycle_tests.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_cycle_tests.csv", index=False, encoding="utf-8-sig")

    index = pd.read_csv(INDEX_FILE)

    print("[info] L4 overtaking event-window analysis")
    window_df, window_summary, window_tests = prepare_overtake_window_features(index)
    window_df.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_window_features.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_window_summary.csv", index=False, encoding="utf-8-sig")
    window_tests.to_csv(OUTPUT_DIR / "4.5_deep_l4_overtake_window_tests.csv", index=False, encoding="utf-8-sig")

    print("[info] L3 space-press window analysis")
    press_df, press_summary, press_tests = prepare_l3_space_press_features(index)
    press_df.to_csv(OUTPUT_DIR / "4.5_deep_l3_space_press_windows.csv", index=False, encoding="utf-8-sig")
    press_summary.to_csv(OUTPUT_DIR / "4.5_deep_l3_space_press_summary.csv", index=False, encoding="utf-8-sig")
    press_tests.to_csv(OUTPUT_DIR / "4.5_deep_l3_space_press_tests.csv", index=False, encoding="utf-8-sig")

    plot_condition_style(
        trial_summary,
        "physio_load_z",
        "Physiological load (within-subject z)",
        OUTPUT_DIR / "4.5_deep_physio_load_condition_style.png",
    )
    plot_condition_style(
        trial_summary,
        "eda_load_z",
        "EDA load (within-subject z)",
        OUTPUT_DIR / "4.5_deep_eda_load_condition_style.png",
    )
    plot_cycle(
        cycle_summary,
        "cycle_eda_load_z",
        "Cycle EDA load (within-subject z)",
        OUTPUT_DIR / "4.5_deep_l4_overtake_cycle_eda_load.png",
    )
    if not window_summary.empty:
        plot_windows(
            window_summary,
            WINDOW_ORDER,
            WINDOW_LABELS,
            "window_eda_load_z",
            "Window EDA load (within-subject z)",
            OUTPUT_DIR / "4.5_deep_l4_overtake_window_eda_load.png",
        )
    if not press_summary.empty:
        plot_windows(
            press_summary,
            PRESS_WINDOW_ORDER,
            PRESS_WINDOW_LABELS,
            "press_eda_load_z",
            "Space-press EDA load (within-subject z)",
            OUTPUT_DIR / "4.5_deep_l3_space_press_eda_load.png",
        )

    write_deep_summary(trial_df, paired, friedman, self_tests, cycle_tests, window_tests, press_df, press_tests)

    print(f"[done] output={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

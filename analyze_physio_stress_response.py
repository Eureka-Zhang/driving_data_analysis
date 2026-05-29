# -*- coding: utf-8 -*-
r"""
Section 4.5: physiological stress response analysis.

Inputs are the per-trial ECG/GSR segments created by preprocess_physio_segments.py.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_physio_stress_response.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASE_DIR = Path(__file__).resolve().parent
SEGMENT_DIR = BASE_DIR / "analysis_output_physio_segments"
INDEX_FILE = SEGMENT_DIR / "trial_time_index.csv"
OUTPUT_DIR = BASE_DIR / "analysis_output_physio_stress"

CONDITION_ORDER = ["l3_follow", "l4_follow", "l4_overtake"]
CONDITION_LABELS = {
    "l3_follow": "L3 Following",
    "l4_follow": "L4 Following",
    "l4_overtake": "L4 Overtaking",
}

STYLE_ORDER = ["aggressive", "consecutive", "neutral", "self"]
STYLE_LABELS = {
    "aggressive": "Aggressive",
    "consecutive": "Conservative",
    "neutral": "Neutral",
    "self": "Self",
}
STYLE_COLORS = {
    "aggressive": "#d62728",
    "consecutive": "#2ca02c",
    "neutral": "#1f77b4",
    "self": "#ff7f0e",
}


@dataclass(frozen=True)
class FeatureMetric:
    key: str
    label: str
    lower_is_load: bool


PLOT_METRICS = [
    FeatureMetric("eda_peak_amp_max", "EDA peak amplitude (uS)", False),
    FeatureMetric("eda_peak_rate_per_min", "EDA peak rate (/min)", False),
    FeatureMetric("rmssd_ms", "RMSSD (ms)", True),
    FeatureMetric("mean_hr_bpm", "Mean heart rate (bpm)", False),
]


def setup_font() -> None:
    if not HAS_MPL:
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<.001"
    return f"{p_value:.3f}"


def style_label(style: str) -> str:
    return STYLE_LABELS.get(style, str(style).title())


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, str(condition))


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def parse_dt_list(value: str) -> list[datetime]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [parse_dt(part) for part in str(value).split(";") if part.strip()]


def sampling_rate(elapsed_s: np.ndarray) -> float:
    diffs = np.diff(elapsed_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return np.nan
    return float(1.0 / np.median(diffs))


def smooth_signal(values: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    if len(values) < 7 or not np.isfinite(fs) or fs <= 0:
        return values
    window = int(round(window_s * fs))
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    if window >= len(values):
        window = len(values) - 1 if len(values) % 2 == 0 else len(values)
    if window < 5:
        return values
    return signal.savgol_filter(values, window_length=window, polyorder=2, mode="interp")


def eda_features_from_arrays(elapsed_s: np.ndarray, gsr: np.ndarray) -> dict:
    valid = np.isfinite(elapsed_s) & np.isfinite(gsr)
    elapsed_s = elapsed_s[valid]
    gsr = gsr[valid]
    if len(gsr) < 20:
        return {
            "eda_duration_s": np.nan,
            "eda_fs_hz": np.nan,
            "eda_mean": np.nan,
            "eda_tonic_mean": np.nan,
            "eda_peak_amp_max": np.nan,
            "eda_peak_count": np.nan,
            "eda_peak_rate_per_min": np.nan,
            "eda_phasic_auc_per_min": np.nan,
        }

    duration = float(elapsed_s[-1] - elapsed_s[0])
    fs = sampling_rate(elapsed_s)
    if not np.isfinite(fs) or fs <= 0:
        return {
            "eda_duration_s": duration,
            "eda_fs_hz": np.nan,
            "eda_mean": float(np.mean(gsr)),
            "eda_tonic_mean": float(np.mean(gsr)),
            "eda_peak_amp_max": np.nan,
            "eda_peak_count": np.nan,
            "eda_peak_rate_per_min": np.nan,
            "eda_phasic_auc_per_min": np.nan,
        }

    gsr_smooth = smooth_signal(gsr, fs, window_s=1.0)
    tonic_window = max(5, int(round(fs * 10.0)))
    tonic = (
        pd.Series(gsr_smooth)
        .rolling(window=tonic_window, min_periods=max(3, tonic_window // 5), center=True)
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )
    if not np.isfinite(tonic).any():
        tonic = np.full_like(gsr_smooth, np.nanmedian(gsr_smooth))
    phasic = gsr_smooth - tonic
    phasic_median = np.nanmedian(phasic)
    if not np.isfinite(phasic_median):
        phasic_median = 0.0
    phasic = phasic - phasic_median

    noise = float(np.nanstd(phasic))
    prominence = max(0.01, 0.5 * noise)
    min_distance = max(1, int(round(fs * 1.0)))
    peaks, properties = signal.find_peaks(phasic, prominence=prominence, distance=min_distance)
    prominences = properties.get("prominences", np.array([], dtype=float))

    positive_phasic = np.maximum(phasic, 0.0)
    dt = 1.0 / fs
    peak_count = int(len(peaks))
    return {
        "eda_duration_s": duration,
        "eda_fs_hz": fs,
        "eda_mean": float(np.mean(gsr_smooth)),
        "eda_tonic_mean": float(np.mean(tonic)),
        "eda_peak_amp_max": float(np.max(prominences)) if len(prominences) else 0.0,
        "eda_peak_count": peak_count,
        "eda_peak_rate_per_min": float(peak_count / duration * 60.0) if duration > 0 else np.nan,
        "eda_phasic_auc_per_min": float(np.sum(positive_phasic) * dt / duration * 60.0) if duration > 0 else np.nan,
    }


def load_gsr_segment(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["elapsed_s", "GSR"])


def eda_features_for_trial(row: pd.Series) -> tuple[dict, list[dict]]:
    gsr_path = Path(str(row["gsr_segment_file"]))
    base = {
        "subject": row["subject"],
        "trial_id": row["trial_id"],
        "condition": row["condition"],
        "style": row["style"],
        "scenario": row["scenario"],
        "duration_s": float(row["duration_s"]),
    }
    cycle_rows: list[dict] = []
    if not gsr_path.exists() or int(row.get("gsr_rows", 0)) <= 0:
        return {**base, **eda_features_from_arrays(np.array([]), np.array([]))}, cycle_rows

    df = load_gsr_segment(gsr_path)
    features = eda_features_from_arrays(df["elapsed_s"].to_numpy(float), df["GSR"].to_numpy(float))

    start_dt = parse_dt(row["start_time"])
    cycle_starts = parse_dt_list(row.get("cycle_starts", ""))
    cycle_ends = parse_dt_list(row.get("cycle_ends", ""))
    for i, (cycle_start, cycle_end) in enumerate(zip(cycle_starts, cycle_ends), start=1):
        start_rel = (cycle_start - start_dt).total_seconds()
        end_rel = (cycle_end - start_dt).total_seconds()
        mask = (df["elapsed_s"] >= start_rel) & (df["elapsed_s"] <= end_rel)
        cycle_features = eda_features_from_arrays(
            df.loc[mask, "elapsed_s"].to_numpy(float),
            df.loc[mask, "GSR"].to_numpy(float),
        )
        cycle_rows.append(
            {
                **base,
                "cycle_index": i,
                "cycle_start_rel_s": start_rel,
                "cycle_end_rel_s": end_rel,
                **cycle_features,
            }
        )

    return {**base, **features}, cycle_rows


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    ecg = ecg[np.isfinite(ecg)]
    if len(ecg) < int(fs * 10):
        return np.array([], dtype=int)

    centered = ecg - np.nanmedian(ecg)
    nyquist = 0.5 * fs
    low = max(0.5 / nyquist, 1e-5)
    high = min(20.0 / nyquist, 0.95)
    try:
        sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos, centered)
    except ValueError:
        filtered = centered

    pos = np.percentile(filtered, 99)
    neg = abs(np.percentile(filtered, 1))
    if neg > pos:
        filtered = -filtered

    spread = np.percentile(filtered, 99) - np.percentile(filtered, 1)
    prominence = max(0.05 * spread, 0.5 * np.std(filtered))
    min_distance = max(1, int(round(0.30 * fs)))
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    return peaks.astype(int)


def hrv_features_for_trial(row: pd.Series) -> dict:
    base = {
        "subject": row["subject"],
        "trial_id": row["trial_id"],
        "condition": row["condition"],
        "style": row["style"],
        "scenario": row["scenario"],
        "duration_s": float(row["duration_s"]),
    }
    ecg_path = Path(str(row["ecg_segment_file"]))
    if not ecg_path.exists() or int(row.get("ecg_rows", 0)) <= 0:
        return {
            **base,
            "ecg_fs_hz": np.nan,
            "r_peak_count": 0,
            "valid_rr_count": 0,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "pnn50": np.nan,
        }

    df = pd.read_csv(ecg_path, usecols=["elapsed_s", "ECG_mV"])
    elapsed = df["elapsed_s"].to_numpy(float)
    ecg = df["ECG_mV"].to_numpy(float)
    fs = sampling_rate(elapsed)
    if not np.isfinite(fs) or fs <= 0:
        return {
            **base,
            "ecg_fs_hz": np.nan,
            "r_peak_count": 0,
            "valid_rr_count": 0,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "pnn50": np.nan,
        }

    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 3:
        rr_s = np.array([], dtype=float)
    else:
        rr_s = np.diff(elapsed[peaks])
        rr_s = rr_s[np.isfinite(rr_s) & (rr_s >= 0.30) & (rr_s <= 2.00)]
        if len(rr_s) >= 3:
            median_rr = np.median(rr_s)
            rr_s = rr_s[(rr_s >= median_rr * 0.5) & (rr_s <= median_rr * 1.5)]

    rr_ms = rr_s * 1000.0
    if len(rr_ms) < 3:
        mean_hr = np.nan
        sdnn = np.nan
        rmssd = np.nan
        pnn50 = np.nan
    else:
        mean_hr = float(60.0 / np.mean(rr_s))
        sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
        rr_diff = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) else np.nan
        pnn50 = float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if len(rr_diff) else np.nan

    return {
        **base,
        "ecg_fs_hz": fs,
        "r_peak_count": int(len(peaks)),
        "valid_rr_count": int(len(rr_ms)),
        "mean_hr_bpm": mean_hr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
    }


def mean_se_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (condition, style), sub in df.groupby(["condition", "style"], sort=False):
        row = {
            "condition": condition,
            "condition_label": condition_label(condition),
            "style": style,
            "style_label": style_label(style),
            "n": int(sub["subject"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    condition_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    style_rank = {s: i for i, s in enumerate(STYLE_ORDER)}
    out["_condition_rank"] = out["condition"].map(condition_rank)
    out["_style_rank"] = out["style"].map(style_rank)
    return out.sort_values(["_condition_rank", "_style_rank"]).drop(columns=["_condition_rank", "_style_rank"])


def paired_test_rows(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    comparisons = [
        ("L3 vs L4 Following", "l3_follow", "l4_follow"),
        ("L4 Overtaking vs L4 Following", "l4_overtake", "l4_follow"),
    ]
    rows = []
    for metric in metrics:
        subject_condition = df.groupby(["subject", "condition"], as_index=False)[metric].mean()
        pivot = subject_condition.pivot(index="subject", columns="condition", values=metric)
        for label, a, b in comparisons:
            if a not in pivot or b not in pivot:
                continue
            paired = pivot[[a, b]].dropna()
            if paired.empty:
                continue
            diff = paired[a] - paired[b]
            if len(diff) > 1 and diff.std(ddof=1) > 0:
                t_stat, t_p = stats.ttest_rel(paired[a], paired[b])
                dz = float(diff.mean() / diff.std(ddof=1))
            else:
                t_stat, t_p, dz = np.nan, np.nan, np.nan
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
                    "p_ttest": float(t_p) if pd.notna(t_p) else np.nan,
                    "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
                    "p_wilcoxon": float(w_p) if pd.notna(w_p) else np.nan,
                    "cohens_dz": dz,
                }
            )
    return pd.DataFrame(rows)


def self_vs_other_rows(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        grouped = df.groupby(["subject", "condition", "style"], as_index=False)[metric].mean()
        self_values = grouped[grouped["style"] == "self"].rename(columns={metric: "self_value"})
        other_values = (
            grouped[grouped["style"] != "self"]
            .groupby(["subject", "condition"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "other_mean"})
        )
        paired = self_values.merge(other_values, on=["subject", "condition"], how="inner")
        for condition, sub in paired.groupby("condition", sort=False):
            diff = sub["self_value"] - sub["other_mean"]
            if len(diff) > 1 and diff.std(ddof=1) > 0:
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


def friedman_style_rows(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
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


def plot_metric(summary: pd.DataFrame, metric: FeatureMetric, out_path: Path) -> None:
    if not HAS_MPL:
        return
    setup_font()
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    x = np.arange(len(CONDITION_ORDER))
    for style in STYLE_ORDER:
        sub = summary[summary["style"] == style].set_index("condition")
        means = [sub.loc[c, f"{metric.key}_mean"] if c in sub.index else np.nan for c in CONDITION_ORDER]
        ses = [sub.loc[c, f"{metric.key}_se"] if c in sub.index else np.nan for c in CONDITION_ORDER]
        ax.errorbar(
            x,
            means,
            yerr=ses,
            marker="o",
            linewidth=2.0,
            capsize=4,
            label=style_label(style),
            color=STYLE_COLORS.get(style),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in CONDITION_ORDER])
    ax.set_ylabel(metric.label)
    ax.set_title(metric.label + " by condition and style")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    feature_df: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    self_tests: pd.DataFrame,
    friedman: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("4.5 生理应激响应分析结果摘要")
    lines.append("")
    lines.append("说明：EDA 使用每个 trial 内的 GSR 信号提取瞬态峰值；HRV 使用 ECG 自动识别 R 峰后计算 RMSSD、SDNN 和平均心率。")
    lines.append("由于当前事件 JSON 中没有跟驰最低车距和超车切出/并入/回道的精确标签，本脚本采用 trial/cycle 窗口进行辅助分析。")
    lines.append("")

    lines.append("一、数据完整性")
    lines.append(f"- 纳入 trial 数：{len(feature_df)}。")
    lines.append(f"- EDA 有效 trial 数：{feature_df['eda_peak_amp_max'].notna().sum()}。")
    lines.append(f"- HRV 有效 trial 数：{feature_df['rmssd_ms'].notna().sum()}。")
    lines.append("")

    lines.append("二、条件层面的辅助趋势")
    for metric in ["eda_peak_amp_max", "eda_peak_rate_per_min", "rmssd_ms", "mean_hr_bpm"]:
        sub = paired[paired["metric"] == metric]
        if sub.empty:
            continue
        lines.append(f"- {metric}:")
        for _, row in sub.iterrows():
            lines.append(
                "  "
                + f"{row['comparison']}，差值={row['mean_diff_a_minus_b']:.4g}，"
                + f"t检验 p={p_text(row['p_ttest'])}，Wilcoxon p={p_text(row['p_wilcoxon'])}。"
            )
    lines.append("")

    lines.append("三、Self 与其他风格")
    for metric in ["eda_peak_amp_max", "rmssd_ms", "mean_hr_bpm"]:
        sub = self_tests[self_tests["metric"] == metric]
        if sub.empty:
            continue
        lines.append(f"- {metric}:")
        for _, row in sub.iterrows():
            lines.append(
                "  "
                + f"{row['condition_label']}，Self-Other={row['self_minus_other']:.4g}，"
                + f"p={p_text(row['p_ttest'])}。"
            )
    lines.append("")

    lines.append("四、风格主效应提示")
    for metric in ["eda_peak_amp_max", "rmssd_ms", "mean_hr_bpm"]:
        sub = friedman[friedman["metric"] == metric]
        if sub.empty:
            continue
        lines.append(f"- {metric}:")
        for _, row in sub.iterrows():
            lines.append(
                "  "
                + f"{row['condition_label']}，Friedman chi2={row['friedman_chi2']:.3f}，"
                + f"p={p_text(row['p_friedman'])}。"
            )
    lines.append("")

    lines.append("写作建议：若 p 值不稳定或不显著，4.5 建议写成辅助证据，不作为 H1-H3 的主要支撑。")

    (OUTPUT_DIR / "4.5_physio_result_summary.txt").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    if not INDEX_FILE.exists():
        print(f"Index file not found: {INDEX_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX_FILE)

    eda_rows: list[dict] = []
    cycle_rows: list[dict] = []
    hrv_rows: list[dict] = []

    for i, row in index.iterrows():
        trial_id = row["trial_id"]
        print(f"[info] {i + 1:03d}/{len(index)} EDA {trial_id}")
        eda, cycles = eda_features_for_trial(row)
        eda_rows.append(eda)
        cycle_rows.extend(cycles)

    for i, row in index.iterrows():
        trial_id = row["trial_id"]
        print(f"[info] {i + 1:03d}/{len(index)} HRV {trial_id}")
        hrv_rows.append(hrv_features_for_trial(row))

    eda_df = pd.DataFrame(eda_rows)
    hrv_df = pd.DataFrame(hrv_rows)
    cycle_df = pd.DataFrame(cycle_rows)

    key_cols = ["subject", "trial_id", "condition", "style", "scenario", "duration_s"]
    feature_df = eda_df.merge(hrv_df, on=key_cols, how="outer")

    metrics = [
        "eda_peak_amp_max",
        "eda_peak_rate_per_min",
        "eda_phasic_auc_per_min",
        "eda_tonic_mean",
        "mean_hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50",
    ]

    summary = mean_se_summary(feature_df, metrics)
    paired = paired_test_rows(feature_df, metrics)
    self_tests = self_vs_other_rows(feature_df, metrics)
    friedman = friedman_style_rows(feature_df, metrics)

    eda_df.to_csv(OUTPUT_DIR / "4.5.1_EDA_trial_features.csv", index=False, encoding="utf-8-sig")
    if not cycle_df.empty:
        cycle_df.to_csv(OUTPUT_DIR / "4.5.1_EDA_cycle_features.csv", index=False, encoding="utf-8-sig")
    hrv_df.to_csv(OUTPUT_DIR / "4.5.2_HRV_trial_features.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUTPUT_DIR / "4.5_physio_trial_features_combined.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "4.5_physio_condition_style_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUTPUT_DIR / "4.5_physio_condition_paired_tests.csv", index=False, encoding="utf-8-sig")
    self_tests.to_csv(OUTPUT_DIR / "4.5_physio_self_vs_other_tests.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(OUTPUT_DIR / "4.5_physio_style_friedman_tests.csv", index=False, encoding="utf-8-sig")

    for metric in PLOT_METRICS:
        plot_metric(summary, metric, OUTPUT_DIR / f"4.5_{metric.key}_condition_style.png")

    write_summary(feature_df, summary, paired, self_tests, friedman)

    print(f"[done] output={OUTPUT_DIR}")
    print(f"[done] trials={len(feature_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

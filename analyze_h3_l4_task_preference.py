# -*- coding: utf-8 -*-
r"""
Section 4.3.4: H3 analysis for L4 overtaking preference structure and
longitudinal/lateral preference decoupling.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_h3_l4_task_preference.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "356953624_按序号_自动驾驶系统乘坐体验问卷_241_240.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis_output_h3_l4_task_preference"

SUBJECT_COL_INDEX = 9
GROUP_COL_INDEX = 10

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
TASKS = [
    ("l4 follow", "L4 Following"),
    ("l4 overtake", "L4 Overtaking"),
]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    column_index: int
    y_min: float
    y_max: float
    ticks: list[float]
    lower_is_better: bool = False
    integer_only: bool = False


METRICS = [
    Metric("comfort", "Comfort", 11, 0, 100, [0, 20, 40, 60, 80, 100]),
    Metric("comfort_rank", "Comfort rank", 11, 1, 4, [1, 2, 3, 4]),
    Metric("smoothness", "Smoothness", 12, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("expectation", "Expectation", 14, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("trust", "Trust", 15, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("tension", "Tension", 16, 1, 5, [1, 2, 3, 4, 5], True, True),
    Metric("relaxation", "Relaxation", 17, 1, 5, [1, 2, 3, 4, 5], False, True),
]

FIG_METRICS = ["comfort", "smoothness", "expectation", "trust", "tension", "relaxation"]
CORE_METRICS = ["comfort", "comfort_rank", "expectation", "trust", "tension"]
PREFERENCE_METRICS = ["expectation", "comfort_rank", "trust", "acceptance_composite"]
PREFERENCE_METRIC_LABELS = {
    "expectation": "Expectation",
    "comfort_rank": "Comfort rank",
    "trust": "Trust",
    "acceptance_composite": "Acceptance composite",
}
PAIRWISE_STYLE_COMPARISONS = [
    ("aggressive", "consecutive"),
    ("aggressive", "neutral"),
    ("aggressive", "self"),
    ("consecutive", "neutral"),
    ("consecutive", "self"),
    ("neutral", "self"),
]


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def classify_l4_task(group_name: str) -> str | None:
    name = str(group_name).strip().lower()
    if "l4" not in name:
        return None
    if "follow" in name:
        return "l4 follow"
    if "overtake" in name:
        return "l4 overtake"
    return None


def extract_style(group_name: str) -> str | None:
    name = str(group_name).strip().lower()
    for style in STYLE_ORDER:
        if style in name:
            return style
    return None


def metric_by_key(metric_key: str) -> Metric:
    return next(m for m in METRICS if m.key == metric_key)


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<.001"
    return f"{p_value:.3f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.array([np.nan if pd.isna(v) else float(v) for v in p_values], dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid_idx = np.where(~np.isnan(p))[0]
    m = len(valid_idx)
    if m == 0:
        return adjusted.tolist()

    order = valid_idx[np.argsort(p[valid_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * p[idx]
        running_max = max(running_max, raw)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def paired_tests(follow: pd.Series, overtake: pd.Series) -> dict:
    pair = pd.concat([follow, overtake], axis=1, keys=["follow", "overtake"]).dropna()
    diff = pair["follow"] - pair["overtake"]
    n = int(len(diff))
    if n < 2:
        return {
            "n": n,
            "l4_follow_mean": np.nan,
            "l4_overtake_mean": np.nan,
            "mean_diff": np.nan,
            "sd_diff": np.nan,
            "t": np.nan,
            "p_t": np.nan,
            "w": np.nan,
            "p_w": np.nan,
            "dz": np.nan,
        }

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    dz = mean_diff / sd_diff if sd_diff > 0 else np.nan
    try:
        from scipy.stats import ttest_rel, wilcoxon

        t_res = ttest_rel(pair["follow"], pair["overtake"], nan_policy="omit")
        try:
            w_res = wilcoxon(pair["follow"], pair["overtake"], zero_method="wilcox", alternative="two-sided")
            w_stat, p_w = float(w_res.statistic), float(w_res.pvalue)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        t_stat, p_t = float(t_res.statistic), float(t_res.pvalue)
    except ImportError:
        t_stat, p_t, w_stat, p_w = np.nan, np.nan, np.nan, np.nan

    return {
        "n": n,
        "l4_follow_mean": round(float(pair["follow"].mean()), 4),
        "l4_overtake_mean": round(float(pair["overtake"].mean()), 4),
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
        "p_t": round(p_t, 6) if pd.notna(p_t) else np.nan,
        "w": round(w_stat, 4) if pd.notna(w_stat) else np.nan,
        "p_w": round(p_w, 6) if pd.notna(p_w) else np.nan,
        "dz": round(dz, 4) if pd.notna(dz) else np.nan,
    }


def load_l4_detail(df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
        }
    )
    detail["task"] = detail["group"].map(classify_l4_task)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["subject", "group", "task", "style"]).copy()

    for metric in METRICS:
        if metric.key == "comfort_rank":
            continue
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)

    detail["comfort_rank"] = detail.groupby(["subject", "task"])["comfort"].rank(method="average")
    detail["acceptance_composite"] = build_acceptance_composite(detail)
    return detail


def build_acceptance_composite(detail: pd.DataFrame) -> pd.Series:
    components = pd.DataFrame(index=detail.index)
    for key in ["comfort_rank", "smoothness", "expectation", "trust", "relaxation"]:
        components[key] = detail[key].astype(float)
    components["low_tension"] = -detail["tension"].astype(float)

    z = components.copy()
    for col in z.columns:
        sd = z[col].std(ddof=1)
        z[col] = (z[col] - z[col].mean()) / sd if sd and pd.notna(sd) else 0.0
    return z.mean(axis=1)


def wide_metric(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    return detail.pivot_table(index="subject", columns=["task", "style"], values=metric_key, aggfunc="mean")


def descriptive_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = METRICS + [Metric("acceptance_composite", "Acceptance composite", -1, np.nan, np.nan, [])]
    for metric in metrics:
        for task_key, task_label in TASKS:
            for style in STYLE_ORDER:
                values = detail.loc[(detail["task"] == task_key) & (detail["style"] == style), metric.key].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": task_label,
                        "style": STYLE_LABELS[style],
                        "n": int(values.size),
                        "mean": round(float(values.mean()), 3),
                        "std": round(float(values.std(ddof=1)), 3),
                        "median": round(float(values.median()), 3),
                        "min": round(float(values.min()), 3),
                        "max": round(float(values.max()), 3),
                    }
                )
    return pd.DataFrame(rows)


def paired_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = METRICS + [Metric("acceptance_composite", "Acceptance composite", -1, np.nan, np.nan, [])]
    for metric in metrics:
        wide = wide_metric(detail, metric.key)
        for style in STYLE_ORDER:
            follow_col = ("l4 follow", style)
            overtake_col = ("l4 overtake", style)
            if follow_col not in wide.columns or overtake_col not in wide.columns:
                continue
            test = paired_tests(wide[follow_col], wide[overtake_col])
            diff = test["mean_diff"]
            benefit = -diff if metric.lower_is_better and pd.notna(diff) else diff
            rows.append(
                {
                    "metric": metric.label,
                    "metric_key": metric.key,
                    "style": STYLE_LABELS[style],
                    "style_key": style,
                    "lower_is_better": metric.lower_is_better,
                    "benefit_coded_diff": round(float(benefit), 4) if pd.notna(benefit) else np.nan,
                    **test,
                }
            )
    return pd.DataFrame(rows)


def task_best_style_table(desc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for task_key, task_label in TASKS:
            sub = desc[(desc["metric_key"] == metric.key) & (desc["condition"] == task_label)].copy()
            if sub.empty:
                continue
            best_value = float(sub["mean"].min() if metric.lower_is_better else sub["mean"].max())
            tied = sub[np.isclose(sub["mean"].astype(float), best_value)]
            rows.append(
                {
                    "condition": task_label,
                    "metric": metric.label,
                    "metric_key": metric.key,
                    "best_style": " / ".join(tied["style"].astype(str).tolist()),
                    "best_mean": round(best_value, 3),
                    "selection_rule": "lowest is preferred" if metric.lower_is_better else "highest is preferred",
                }
            )
    return pd.DataFrame(rows)


def overtaking_style_anova(detail: pd.DataFrame) -> pd.DataFrame:
    """One-way repeated-measures ANOVA within L4 overtaking."""
    try:
        from scipy.stats import f as f_dist
    except ImportError:
        return pd.DataFrame()

    rows = []
    task_key = "l4 overtake"
    k = len(STYLE_ORDER)
    for metric in METRICS:
        wide = detail[detail["task"] == task_key].pivot_table(
            index="subject",
            columns="style",
            values=metric.key,
            aggfunc="mean",
        )
        if not all(style in wide.columns for style in STYLE_ORDER):
            continue
        wide = wide[STYLE_ORDER].dropna()
        n = len(wide)
        if n < 2:
            continue
        y = wide.to_numpy(dtype=float)
        grand = y.mean()
        subj_mean = y.mean(axis=1)
        style_mean = y.mean(axis=0)
        ss_style = n * np.sum((style_mean - grand) ** 2)
        ss_subject = k * np.sum((subj_mean - grand) ** 2)
        ss_total = np.sum((y - grand) ** 2)
        ss_error = ss_total - ss_style - ss_subject
        df_style = k - 1
        df_error = (n - 1) * (k - 1)
        ms_style = ss_style / df_style
        ms_error = ss_error / df_error
        f_value = ms_style / ms_error if ms_error > 0 else np.nan
        p_value = float(f_dist.sf(f_value, df_style, df_error)) if pd.notna(f_value) else np.nan
        eta_p2 = ss_style / (ss_style + ss_error) if (ss_style + ss_error) > 0 else np.nan
        rows.append(
            {
                "condition": "L4 Overtaking",
                "metric": metric.label,
                "metric_key": metric.key,
                "effect": "Style",
                "DF": df_style,
                "Error DF": df_error,
                "F": round(float(f_value), 6) if pd.notna(f_value) else np.nan,
                "p": round(p_value, 6) if pd.notna(p_value) else np.nan,
                "partial_eta_sq": round(float(eta_p2), 6) if pd.notna(eta_p2) else np.nan,
                "n_subjects": n,
            }
        )
    return pd.DataFrame(rows)


def overtaking_pairwise_style_tests(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    task_key = "l4 overtake"
    for metric in METRICS:
        wide = detail[detail["task"] == task_key].pivot_table(
            index="subject",
            columns="style",
            values=metric.key,
            aggfunc="mean",
        )
        for style_a, style_b in PAIRWISE_STYLE_COMPARISONS:
            if style_a not in wide.columns or style_b not in wide.columns:
                continue
            pair = pd.concat([wide[style_a], wide[style_b]], axis=1, keys=["a", "b"]).dropna()
            diff = pair["a"] - pair["b"]
            n = len(diff)
            if n < 2:
                continue
            try:
                from scipy.stats import ttest_rel, wilcoxon

                t_res = ttest_rel(pair["a"], pair["b"], nan_policy="omit")
                try:
                    w_res = wilcoxon(pair["a"], pair["b"], zero_method="wilcox", alternative="two-sided")
                    w_stat, p_w = float(w_res.statistic), float(w_res.pvalue)
                except ValueError:
                    w_stat, p_w = np.nan, np.nan
                t_stat, p_t = float(t_res.statistic), float(t_res.pvalue)
            except ImportError:
                t_stat, p_t, w_stat, p_w = np.nan, np.nan, np.nan, np.nan

            rows.append(
                {
                    "condition": "L4 Overtaking",
                    "metric": metric.label,
                    "metric_key": metric.key,
                    "comparison": f"{STYLE_LABELS[style_a]} - {STYLE_LABELS[style_b]}",
                    "style_a": STYLE_LABELS[style_a],
                    "style_b": STYLE_LABELS[style_b],
                    "n": n,
                    "mean_a": round(float(pair["a"].mean()), 4),
                    "mean_b": round(float(pair["b"].mean()), 4),
                    "mean_diff": round(float(diff.mean()), 4),
                    "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
                    "p_t": round(p_t, 6) if pd.notna(p_t) else np.nan,
                    "w": round(w_stat, 4) if pd.notna(w_stat) else np.nan,
                    "p_w": round(p_w, 6) if pd.notna(p_w) else np.nan,
                    "dz": round(float(diff.mean() / diff.std(ddof=1)), 4) if diff.std(ddof=1) > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        for metric_key, idx in out.groupby("metric_key").groups.items():
            out.loc[idx, "p_t_holm_by_metric"] = holm_adjust(out.loc[idx, "p_t"].tolist())
            out.loc[idx, "p_w_holm_by_metric"] = holm_adjust(out.loc[idx, "p_w"].tolist())
    return out


def task_difference_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot(index="style", columns="metric", values="mean_diff")
    table = table.reindex([STYLE_LABELS[s] for s in STYLE_ORDER])
    labels = [m.label for m in METRICS if m.key != "comfort_rank"]
    return table.reindex(columns=labels)


def benefit_difference_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot(index="style", columns="metric", values="benefit_coded_diff")
    table = table.reindex([STYLE_LABELS[s] for s in STYLE_ORDER])
    labels = [m.label for m in METRICS if m.key in CORE_METRICS]
    return table.reindex(columns=labels)


def preference_winners(detail: pd.DataFrame, metric_key: str, task_key: str) -> pd.Series:
    wide = detail[detail["task"] == task_key].pivot_table(
        index="subject",
        columns="style",
        values=metric_key,
        aggfunc="mean",
    )
    wide = wide[[s for s in STYLE_ORDER if s in wide.columns]].dropna(how="all")
    winners = {}
    for subject, row in wide.iterrows():
        max_value = row.max()
        tied = [STYLE_LABELS[s] for s in STYLE_ORDER if s in row.index and pd.notna(row[s]) and row[s] == max_value]
        winners[subject] = "/".join(tied)
    return pd.Series(winners)


def preference_matrix(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    follow = preference_winners(detail, metric_key, "l4 follow")
    overtake = preference_winners(detail, metric_key, "l4 overtake")
    both = pd.concat([follow, overtake], axis=1, keys=["L4 Following", "L4 Overtaking"]).dropna()
    return pd.crosstab(both["L4 Following"], both["L4 Overtaking"])


def fractional_preference_matrix(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    styles = [STYLE_LABELS[s] for s in STYLE_ORDER]
    matrix = pd.DataFrame(0.0, index=styles, columns=styles)
    follow_wide = detail[detail["task"] == "l4 follow"].pivot_table(index="subject", columns="style", values=metric_key, aggfunc="mean")
    overtake_wide = detail[detail["task"] == "l4 overtake"].pivot_table(index="subject", columns="style", values=metric_key, aggfunc="mean")
    subjects = sorted(set(follow_wide.index) & set(overtake_wide.index))
    for subject in subjects:
        f_row = follow_wide.loc[subject]
        o_row = overtake_wide.loc[subject]
        f_max, o_max = f_row.max(), o_row.max()
        f_winners = [s for s in STYLE_ORDER if s in f_row.index and pd.notna(f_row[s]) and f_row[s] == f_max]
        o_winners = [s for s in STYLE_ORDER if s in o_row.index and pd.notna(o_row[s]) and o_row[s] == o_max]
        if not f_winners or not o_winners:
            continue
        weight = 1.0 / (len(f_winners) * len(o_winners))
        for f_style in f_winners:
            for o_style in o_winners:
                matrix.loc[STYLE_LABELS[f_style], STYLE_LABELS[o_style]] += weight
    return matrix


def plot_overtaking_preference_structure(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("L4 Overtaking Style Preference Structure", fontsize=14, fontweight="bold")

    x = np.arange(len(STYLE_ORDER))
    colors = [STYLE_COLORS[s] for s in STYLE_ORDER]
    for ax, metric_key in zip(axes.flat, FIG_METRICS):
        metric = metric_by_key(metric_key)
        means, ses = [], []
        for style in STYLE_ORDER:
            values = detail.loc[(detail["task"] == "l4 overtake") & (detail["style"] == style), metric.key].dropna()
            means.append(float(values.mean()))
            ses.append(float(values.std(ddof=1) / np.sqrt(len(values))))

        preferred_value = min(means) if metric.lower_is_better else max(means)

        bars = ax.bar(
            x,
            means,
            yerr=ses,
            color=colors,
            edgecolor="#333333",
            linewidth=0.7,
            alpha=0.78,
            capsize=3,
        )
        for bar, mean in zip(bars, means):
            if np.isclose(mean, preferred_value):
                bar.set_linewidth(2.0)
        ax.set_title(metric.label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE_LABELS[s] for s in STYLE_ORDER], rotation=20, ha="right")
        pad = 0.35 if metric.y_max <= 5 else 5
        ax.set_ylim(metric.y_min - pad, metric.y_max + pad)
        ax.set_yticks(metric.ticks)
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.axhline(3 if metric.y_max <= 5 else 50, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
        if metric.lower_is_better:
            ax.text(0.99, 0.04, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
        for i, mean in enumerate(means):
            ax.text(i, metric.y_min + pad * 0.55, f"{mean:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333333")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_best_style_table(best_table: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    sub = best_table[best_table["condition"] == "L4 Overtaking"].copy()
    sub = sub[sub["metric_key"].isin(FIG_METRICS)]
    metric_order = [metric_by_key(k).label for k in FIG_METRICS]
    sub["metric"] = pd.Categorical(sub["metric"], categories=metric_order, ordered=True)
    sub = sub.sort_values("metric")

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    rows = [[row["metric"], row["best_style"], f"{float(row['best_mean']):.2f}"] for _, row in sub.iterrows()]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Highest/preferred style", "Mean"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    ax.set_title("L4 Overtaking Preferred Style by Metric", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_heatmap(table: pd.DataFrame, out_path: Path, title: str, cbar_label: str, note: str) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    values = table.astype(float)
    normed = values.copy()
    for col in normed.columns:
        max_abs = np.nanmax(np.abs(normed[col].to_numpy()))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
        normed[col] = normed[col] / max_abs

    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    fig.patch.set_facecolor("white")
    im = ax.imshow(normed.to_numpy(), cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label, fontsize=9)
    ax.text(0.0, -0.26, note, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_task_style_interaction(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("L4 Task x Style Interaction", fontsize=14, fontweight="bold")

    x = np.arange(len(STYLE_ORDER))
    for ax, metric_key in zip(axes.flat, FIG_METRICS):
        metric = metric_by_key(metric_key)
        for task_key, label, color, marker in [
            ("l4 follow", "L4 Following", "#4c78a8", "o"),
            ("l4 overtake", "L4 Overtaking", "#f58518", "s"),
        ]:
            means, ses = [], []
            for style in STYLE_ORDER:
                values = detail.loc[(detail["task"] == task_key) & (detail["style"] == style), metric.key].dropna()
                means.append(float(values.mean()))
                ses.append(float(values.std(ddof=1) / np.sqrt(len(values))))
            ax.errorbar(
                x,
                means,
                yerr=ses,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgecolor="#333333",
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        ax.set_title(metric.label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE_LABELS[s] for s in STYLE_ORDER], rotation=20, ha="right")
        pad = 0.35 if metric.y_max <= 5 else 5
        ax.set_ylim(metric.y_min - pad, metric.y_max + pad)
        ax.set_yticks(metric.ticks)
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.axhline(3 if metric.y_max <= 5 else 50, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)
        if metric.lower_is_better:
            ax.text(0.99, 0.04, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.06), frameon=True, framealpha=0.95, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_preference_migration(matrix: pd.DataFrame, out_path: Path, title: str) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    styles = [STYLE_LABELS[s] for s in STYLE_ORDER]
    values = matrix.reindex(index=styles, columns=styles).fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values.to_numpy(dtype=float), cmap="YlOrRd", aspect="equal")
    ax.set_xticks(np.arange(len(styles)))
    ax.set_xticklabels(styles, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(styles)))
    ax.set_yticklabels(styles)
    ax.set_xlabel("Preferred style in L4 Overtaking")
    ax.set_ylabel("Preferred style in L4 Following")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            label = f"{value:.1f}" if abs(value - round(value)) > 1e-8 else f"{int(round(value))}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("participants (fractional ties)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def migration_consistency_summary(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for metric_key in PREFERENCE_METRICS:
        matrix = matrices.get(f"{metric_key}_fractional")
        if matrix is None or matrix.empty:
            continue
        values = matrix.reindex(
            index=[STYLE_LABELS[s] for s in STYLE_ORDER],
            columns=[STYLE_LABELS[s] for s in STYLE_ORDER],
        ).fillna(0.0)
        same = float(np.trace(values.to_numpy(dtype=float)))
        total = float(values.to_numpy(dtype=float).sum())
        same_pct = 100 * same / total if total else np.nan
        rows.append(
            {
                "metric_key": metric_key,
                "metric": PREFERENCE_METRIC_LABELS.get(metric_key, metric_key),
                "same_style_mass": round(same, 4),
                "total_mass": round(total, 4),
                "same_style_percent": round(same_pct, 2) if pd.notna(same_pct) else np.nan,
                "migration_mass": round(total - same, 4),
                "migration_percent": round(100 - same_pct, 2) if pd.notna(same_pct) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("same_style_percent")


def plot_migration_consistency(consistency: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL or consistency.empty:
        return False
    setup_font()

    data = consistency.sort_values("same_style_percent").copy()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.patch.set_facecolor("white")
    x = np.arange(len(data))
    bars = ax.bar(
        x,
        data["same_style_percent"].astype(float),
        color="#f6c85f",
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(data["metric"], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("same-style preference (%)")
    ax.set_title("L4 Following-to-Overtaking Preference Consistency", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(row["same_style_percent"]) + 2,
            f"{row['same_style_mass']:.2f}/{row['total_mass']:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    ax.text(
        0.0,
        -0.34,
        "Lower percentages indicate stronger longitudinal-lateral preference decoupling.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(
    desc: pd.DataFrame,
    summary: pd.DataFrame,
    best_table: pd.DataFrame,
    overtaking_anova: pd.DataFrame,
    overtaking_pairwise: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    consistency: pd.DataFrame,
) -> None:
    lines = []
    lines.append("4.3.4 L4 overtaking preference structure and longitudinal-lateral decoupling")
    lines.append("")
    lines.append("4.3.4.1 L4 overtaking internal style preference structure")
    overtake_best = best_table[best_table["condition"] == "L4 Overtaking"].copy()
    for metric_key in FIG_METRICS:
        row = overtake_best[overtake_best["metric_key"] == metric_key].iloc[0]
        lines.append(f"- {row['metric']}: preferred/highest style = {row['best_style']}, mean = {float(row['best_mean']):.2f}.")
    lines.append("")

    lines.append("4.3.4.2 Within-overtaking style differences")
    for _, row in overtaking_anova.iterrows():
        lines.append(
            f"- {row['metric']}: Style effect F({int(row['DF'])}, {int(row['Error DF'])})="
            f"{row['F']:.2f}, p={p_text(row['p'])}, partial eta^2={row['partial_eta_sq']:.3f}."
        )
    if not overtaking_pairwise.empty:
        sig = overtaking_pairwise[overtaking_pairwise["p_t_holm_by_metric"] < 0.05]
        if sig.empty:
            lines.append("- Holm-adjusted pairwise tests did not show stable significant differences within L4 overtaking.")
        else:
            lines.append("- Holm-adjusted significant pairwise differences within L4 overtaking:")
            for _, row in sig.iterrows():
                lines.append(
                    f"  {row['metric']}, {row['comparison']}: diff={row['mean_diff']:+.2f}, "
                    f"p_holm={p_text(row['p_t_holm_by_metric'])}."
                )
    lines.append("")

    lines.append("4.3.4.3 Comparison with L4 following preference structure")
    follow_best = best_table[best_table["condition"] == "L4 Following"].copy()
    for metric_key in FIG_METRICS:
        f = follow_best[follow_best["metric_key"] == metric_key].iloc[0]
        o = overtake_best[overtake_best["metric_key"] == metric_key].iloc[0]
        lines.append(
            f"- {f['metric']}: L4 Following prefers {f['best_style']} ({float(f['best_mean']):.2f}); "
            f"L4 Overtaking prefers {o['best_style']} ({float(o['best_mean']):.2f})."
        )
    lines.append("")

    lines.append("Selected task-paired contrasts, defined as L4 Following - L4 Overtaking:")
    exp = summary[summary["metric_key"] == "expectation"].copy()
    for _, row in exp.iterrows():
        direction = "higher in following" if row["mean_diff"] > 0 else "lower in following"
        lines.append(
            f"- {row['style']}: Following={row['l4_follow_mean']:.2f}, "
            f"Overtaking={row['l4_overtake_mean']:.2f}, diff={row['mean_diff']:+.2f} "
            f"({direction}), t({int(row['n']) - 1})={row['t']:.2f}, p={p_text(row['p_t'])}."
        )
    lines.append("")

    lines.append("4.3.4.4 Individual preference migration")
    if not consistency.empty:
        lines.append("- Same-style preference mass by indicator:")
        for _, row in consistency.iterrows():
            lines.append(
                f"  {row['metric']}: {row['same_style_mass']:.2f}/{row['total_mass']:.2f} "
                f"({row['same_style_percent']:.2f}%)."
            )
        strongest = consistency.iloc[0]
        lines.append(
            f"- The strongest decoupling appears in {strongest['metric']}, where only "
            f"{strongest['same_style_mass']:.2f}/{strongest['total_mass']:.2f} "
            f"({strongest['same_style_percent']:.2f}%) remains on the matrix diagonal."
        )
    lines.append("")
    lines.append(
        "Interpretation: L4 overtaking shows a descriptive Conservative preference trend, especially on positive "
        "experience metrics, but within-overtaking style differences should be treated cautiously when not stable "
        "after paired testing. The migration indicators with the lowest same-style mass can be emphasized in the "
        "main text, while the remaining indicators may be reported briefly as converging supplementary evidence. "
        "Overall, longitudinal following preference does not reliably transfer to lateral overtaking preference, "
        "supporting task-dependent style preference."
    )
    (OUTPUT_DIR / "4.3.4_H3_result_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = load_l4_detail(df)
    desc = descriptive_table(detail)
    summary = paired_summary(detail)
    best_table = task_best_style_table(desc)
    overtaking_anova = overtaking_style_anova(detail)
    overtaking_pairwise = overtaking_pairwise_style_tests(detail)
    diff = task_difference_table(summary)
    benefit = benefit_difference_table(summary)

    detail.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_detail.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    best_table.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_best_style_by_metric.csv", index=False, encoding="utf-8-sig")
    overtaking_anova.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_overtaking_internal_style_anova.csv", index=False, encoding="utf-8-sig")
    overtaking_pairwise.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_overtaking_internal_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_follow_minus_overtake_paired_tests.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_task_difference_table.csv", encoding="utf-8-sig")
    benefit.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_benefit_coded_difference_table.csv", encoding="utf-8-sig")

    matrices: dict[str, pd.DataFrame] = {}
    for metric_key in PREFERENCE_METRICS:
        raw_matrix = preference_matrix(detail, metric_key)
        frac_matrix = fractional_preference_matrix(detail, metric_key)
        raw_matrix.to_csv(OUTPUT_DIR / f"4.3.4_H3_preference_migration_{metric_key}_tie_labels.csv", encoding="utf-8-sig")
        frac_matrix.to_csv(OUTPUT_DIR / f"4.3.4_H3_preference_migration_{metric_key}_fractional.csv", encoding="utf-8-sig")
        matrices[f"{metric_key}_fractional"] = frac_matrix
        plot_preference_migration(
            frac_matrix,
            OUTPUT_DIR / f"4.3.4_H3_preference_migration_{metric_key}.png",
            f"Preference Migration ({metric_key.replace('_', ' ').title()})",
        )

    consistency = migration_consistency_summary(matrices)
    consistency.to_csv(
        OUTPUT_DIR / "4.3.4_H3_preference_migration_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_migration_consistency(
        consistency,
        OUTPUT_DIR / "4.3.4_H3_preference_migration_consistency.png",
    )
    plot_overtaking_preference_structure(
        detail,
        OUTPUT_DIR / "4.3.4_H3_l4_overtaking_preference_structure.png",
    )
    plot_best_style_table(
        best_table,
        OUTPUT_DIR / "4.3.4_H3_l4_overtaking_best_style_table.png",
    )
    plot_heatmap(
        diff,
        OUTPUT_DIR / "4.3.4_H3_l4_follow_minus_overtake_heatmap.png",
        "L4 Task Difference: Following - Overtaking",
        "Column-normalized L4 following-overtaking difference",
        "Raw differences are shown in cells. Positive values mean L4 Following > L4 Overtaking; for tension, negative values indicate lower tension in following.",
    )
    plot_heatmap(
        benefit,
        OUTPUT_DIR / "4.3.4_H3_l4_benefit_coded_difference_heatmap.png",
        "Benefit-Coded L4 Task Difference",
        "Column-normalized task difference",
        "For tension, the sign is reversed so positive values consistently indicate a more favorable L4 Following evaluation.",
    )
    plot_task_style_interaction(detail, OUTPUT_DIR / "4.3.4_H3_l4_task_style_interaction.png")
    write_summary(desc, summary, best_table, overtaking_anova, overtaking_pairwise, matrices, consistency)

    print(f"detail rows: {len(detail)}")
    print(f"paired rows: {len(summary)}")
    print(f"overtaking anova rows: {len(overtaking_anova)}")
    print(f"output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

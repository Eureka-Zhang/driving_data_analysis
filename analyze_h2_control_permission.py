# -*- coding: utf-8 -*-
r"""
Section 4.3.3: control permission effects on following-style acceptance.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_h2_control_permission.py
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
OUTPUT_DIR = BASE_DIR / "analysis_output_h2_control_permission"

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


@dataclass(frozen=True)
class Metric:
    key: str
    label_cn: str
    label_en: str
    column_index: int
    y_min: float
    y_max: float
    ticks: list[float]
    lower_is_better: bool = False
    integer_only: bool = False


METRICS = [
    Metric("comfort", "舒适度", "Comfort", 11, 0, 100, [0, 20, 40, 60, 80, 100]),
    Metric("comfort_rank", "舒适度排名", "Comfort rank", 11, 1, 4, [1, 2, 3, 4]),
    Metric("smoothness", "平稳性", "Smoothness", 12, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("expectation", "预期一致性", "Expectation", 14, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("trust", "信任度", "Trust", 15, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("tension", "紧张感", "Tension", 16, 1, 5, [1, 2, 3, 4, 5], True, True),
    Metric("relaxation", "放松感", "Relaxation", 17, 1, 5, [1, 2, 3, 4, 5], False, True),
]

CORE_METRICS = ["comfort", "comfort_rank", "expectation", "trust", "tension"]
FIG_METRICS = ["comfort", "smoothness", "expectation", "trust", "tension", "relaxation"]
HEATMAP_METRIC_LABELS_EN = {
    "舒适度": "Comfort",
    "舒适度排名": "Comfort rank",
    "平稳性": "Smoothness",
    "预期一致性": "Expectation",
    "信任度": "Trust",
    "紧张感": "Tension",
    "放松感": "Relaxation",
}


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def classify_follow_permission(group_name: str) -> str | None:
    name = str(group_name).strip().lower()
    if name.startswith("l3"):
        return "L3 Following"
    if "l4" in name and "follow" in name:
        return "L4 Following"
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


def paired_tests(l3: pd.Series, l4: pd.Series) -> dict:
    pair = pd.concat([l3, l4], axis=1, keys=["l3", "l4"]).dropna()
    diff = pair["l3"] - pair["l4"]
    n = int(len(diff))
    if n < 2:
        return {
            "n": n,
            "l3_mean": np.nan,
            "l4_mean": np.nan,
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

        t_res = ttest_rel(pair["l3"], pair["l4"], nan_policy="omit")
        try:
            w_res = wilcoxon(pair["l3"], pair["l4"], zero_method="wilcox", alternative="two-sided")
            w_stat, p_w = float(w_res.statistic), float(w_res.pvalue)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        t_stat, p_t = float(t_res.statistic), float(t_res.pvalue)
    except ImportError:
        t_stat, p_t, w_stat, p_w = np.nan, np.nan, np.nan, np.nan

    return {
        "n": n,
        "l3_mean": round(float(pair["l3"].mean()), 4),
        "l4_mean": round(float(pair["l4"].mean()), 4),
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
        "p_t": round(p_t, 6) if pd.notna(p_t) else np.nan,
        "w": round(w_stat, 4) if pd.notna(w_stat) else np.nan,
        "p_w": round(p_w, 6) if pd.notna(p_w) else np.nan,
        "dz": round(dz, 4) if pd.notna(dz) else np.nan,
    }


def load_following_detail(df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
        }
    )
    detail["permission"] = detail["group"].map(classify_follow_permission)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["subject", "group", "permission", "style"]).copy()

    for metric in METRICS:
        if metric.key == "comfort_rank":
            continue
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)

    detail["comfort_rank"] = detail.groupby(["subject", "permission"])["comfort"].rank(method="average")
    return detail


def wide_metric(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    return detail.pivot_table(
        index="subject",
        columns=["permission", "style"],
        values=metric_key,
        aggfunc="mean",
    )


def paired_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        wide = wide_metric(detail, metric.key)
        for style in STYLE_ORDER:
            l3_col = ("L3 Following", style)
            l4_col = ("L4 Following", style)
            if l3_col not in wide.columns or l4_col not in wide.columns:
                continue
            test = paired_tests(wide[l3_col], wide[l4_col])
            diff = test["mean_diff"]
            benefit = -diff if metric.lower_is_better and pd.notna(diff) else diff
            rows.append(
                {
                    "metric": metric.label_cn,
                    "metric_en": metric.label_en,
                    "metric_key": metric.key,
                    "style": STYLE_LABELS[style],
                    "style_key": style,
                    "lower_is_better": metric.lower_is_better,
                    "benefit_coded_diff": round(float(benefit), 4) if pd.notna(benefit) else np.nan,
                    **test,
                }
            )

    out = pd.DataFrame(rows)
    for metric_key, idx in out.groupby("metric_key").groups.items():
        out.loc[idx, "p_t_holm_by_metric"] = holm_adjust(out.loc[idx, "p_t"].tolist())
        out.loc[idx, "p_w_holm_by_metric"] = holm_adjust(out.loc[idx, "p_w"].tolist())
    return out


def control_gain_wide(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot(index="style", columns="metric", values="mean_diff")
    style_labels = [STYLE_LABELS[s] for s in STYLE_ORDER]
    table = table.reindex(style_labels)
    metric_labels = [m.label_cn for m in METRICS if m.key != "comfort_rank"]
    return table.reindex(columns=metric_labels)


def benefit_gain_wide(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot(index="style", columns="metric", values="benefit_coded_diff")
    style_labels = [STYLE_LABELS[s] for s in STYLE_ORDER]
    table = table.reindex(style_labels)
    metric_labels = [m.label_cn for m in METRICS if m.key in CORE_METRICS]
    return table.reindex(columns=metric_labels)


def descriptive_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for permission in ["L3 Following", "L4 Following"]:
            for style in STYLE_ORDER:
                values = detail.loc[
                    (detail["permission"] == permission) & (detail["style"] == style),
                    metric.key,
                ].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "metric": metric.label_cn,
                        "metric_en": metric.label_en,
                        "metric_key": metric.key,
                        "condition": permission,
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


def rm_anova_tables(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    try:
        from statsmodels.stats.anova import AnovaRM
    except ImportError:
        return manual_rm_anova_tables(detail)

    for metric in METRICS:
        sub = detail[["subject", "permission", "style", metric.key]].dropna().copy()
        sub = sub.rename(columns={metric.key: "score"})
        counts = sub.groupby("subject").size()
        complete_subjects = counts[counts == 8].index
        sub = sub[sub["subject"].isin(complete_subjects)].copy()
        if sub["subject"].nunique() < 2:
            continue
        try:
            fit = AnovaRM(sub, depvar="score", subject="subject", within=["permission", "style"]).fit()
        except Exception:
            continue
        table = fit.anova_table.reset_index().rename(columns={"index": "effect"})
        tables[metric.key] = table
    return tables


def manual_rm_anova_tables(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Balanced 2 x 4 repeated-measures ANOVA without external dependencies."""
    tables: dict[str, pd.DataFrame] = {}
    try:
        from scipy.stats import f as f_dist
    except ImportError:
        return tables

    permissions = ["L3 Following", "L4 Following"]
    styles = STYLE_ORDER
    a, b = len(permissions), len(styles)

    for metric in METRICS:
        wide = detail.pivot_table(
            index="subject",
            columns=["permission", "style"],
            values=metric.key,
            aggfunc="mean",
        )
        cols = [(p, s) for p in permissions for s in styles]
        if not all(col in wide.columns for col in cols):
            continue
        wide = wide[cols].dropna()
        n = len(wide)
        if n < 2:
            continue

        y = wide.to_numpy(dtype=float).reshape(n, a, b)
        grand = y.mean()
        subj_mean = y.mean(axis=(1, 2))
        a_mean = y.mean(axis=(0, 2))
        b_mean = y.mean(axis=(0, 1))
        ab_mean = y.mean(axis=0)
        subj_a_mean = y.mean(axis=2)
        subj_b_mean = y.mean(axis=1)

        ss_a = b * n * np.sum((a_mean - grand) ** 2)
        ss_b = a * n * np.sum((b_mean - grand) ** 2)
        ss_ab = n * np.sum((ab_mean - a_mean[:, None] - b_mean[None, :] + grand) ** 2)
        ss_sxa = b * np.sum((subj_a_mean - subj_mean[:, None] - a_mean[None, :] + grand) ** 2)
        ss_sxb = a * np.sum((subj_b_mean - subj_mean[:, None] - b_mean[None, :] + grand) ** 2)
        ss_sxab = np.sum(
            (
                y
                - subj_a_mean[:, :, None]
                - subj_b_mean[:, None, :]
                - ab_mean[None, :, :]
                + subj_mean[:, None, None]
                + a_mean[None, :, None]
                + b_mean[None, None, :]
                - grand
            )
            ** 2
        )

        rows = []
        specs = [
            ("Permission", ss_a, a - 1, ss_sxa, (n - 1) * (a - 1)),
            ("Style", ss_b, b - 1, ss_sxb, (n - 1) * (b - 1)),
            ("Permission:Style", ss_ab, (a - 1) * (b - 1), ss_sxab, (n - 1) * (a - 1) * (b - 1)),
        ]
        for effect, ss, df_num, ss_err, df_den in specs:
            ms = ss / df_num
            ms_err = ss_err / df_den
            f_value = ms / ms_err if ms_err > 0 else np.nan
            p_value = float(f_dist.sf(f_value, df_num, df_den)) if pd.notna(f_value) else np.nan
            eta_p2 = ss / (ss + ss_err) if (ss + ss_err) > 0 else np.nan
            rows.append(
                {
                    "effect": effect,
                    "SS": round(float(ss), 6),
                    "DF": df_num,
                    "MS": round(float(ms), 6),
                    "Error SS": round(float(ss_err), 6),
                    "Error DF": df_den,
                    "Error MS": round(float(ms_err), 6),
                    "F": round(float(f_value), 6) if pd.notna(f_value) else np.nan,
                    "p": round(p_value, 6) if pd.notna(p_value) else np.nan,
                    "partial_eta_sq": round(float(eta_p2), 6) if pd.notna(eta_p2) else np.nan,
                    "method": "manual_balanced_rm_anova",
                }
            )
        tables[metric.key] = pd.DataFrame(rows)
    return tables


def plot_gain_heatmap(table: pd.DataFrame, out_path: Path, title: str, note: str) -> bool:
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

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    fig.patch.set_facecolor("white")
    im = ax.imshow(normed.to_numpy(), cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels([HEATMAP_METRIC_LABELS_EN.get(c, c) for c in values.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            if pd.isna(value):
                text = ""
            elif abs(value) >= 10:
                text = f"{value:+.2f}"
            else:
                text = f"{value:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="#222222")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Column-normalized L3-L4 difference", fontsize=9)
    ax.text(0.0, -0.24, note, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_permission_style_interaction(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Permission x Style Interaction in Following Scenarios", fontsize=14, fontweight="bold")

    x = np.arange(len(STYLE_ORDER))
    for ax, metric_key in zip(axes.flat, FIG_METRICS):
        metric = metric_by_key(metric_key)
        for permission, color, marker in [
            ("L3 Following", "#4c78a8", "o"),
            ("L4 Following", "#f58518", "s"),
        ]:
            means, ses = [], []
            for style in STYLE_ORDER:
                values = detail.loc[
                    (detail["permission"] == permission) & (detail["style"] == style),
                    metric.key,
                ].dropna()
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
                label=permission,
            )

        ax.set_title(metric.label_en, fontsize=11, fontweight="bold")
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


def plot_following_means(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("L3 vs L4 Following: Mean Ratings by Style", fontsize=14, fontweight="bold")

    width = 0.36
    x = np.arange(len(STYLE_ORDER))
    for ax, metric_key in zip(axes.flat, FIG_METRICS):
        metric = metric_by_key(metric_key)
        l3_means, l4_means, l3_se, l4_se = [], [], [], []
        for style in STYLE_ORDER:
            l3 = detail.loc[
                (detail["permission"] == "L3 Following") & (detail["style"] == style),
                metric.key,
            ].dropna()
            l4 = detail.loc[
                (detail["permission"] == "L4 Following") & (detail["style"] == style),
                metric.key,
            ].dropna()
            l3_means.append(float(l3.mean()))
            l4_means.append(float(l4.mean()))
            l3_se.append(float(l3.std(ddof=1) / np.sqrt(len(l3))))
            l4_se.append(float(l4.std(ddof=1) / np.sqrt(len(l4))))

        ax.bar(x - width / 2, l3_means, width, yerr=l3_se, label="L3 Following", color="#4c78a8", alpha=0.82, capsize=3)
        ax.bar(x + width / 2, l4_means, width, yerr=l4_se, label="L4 Following", color="#f58518", alpha=0.82, capsize=3)
        ax.set_title(metric.label_en, fontsize=11, fontweight="bold")
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
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=True, framealpha=0.95, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_aggressive_pairs(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Aggressive Following: Paired L3-L4 Changes", fontsize=14, fontweight="bold")

    sub = detail[detail["style"] == "aggressive"].copy()
    for ax, metric_key in zip(axes.flat, FIG_METRICS):
        metric = metric_by_key(metric_key)
        wide = sub.pivot_table(index="subject", columns="permission", values=metric.key, aggfunc="mean")
        wide = wide.dropna(subset=["L3 Following", "L4 Following"])

        for _, row in wide.iterrows():
            ax.plot([0, 1], [row["L3 Following"], row["L4 Following"]], color="#999999", alpha=0.45, linewidth=0.8)
        means = [float(wide["L3 Following"].mean()), float(wide["L4 Following"].mean())]
        ax.plot([0, 1], means, color="#d62728", marker="D", markerfacecolor="white", markeredgecolor="#333333", linewidth=2.0)
        ax.scatter(np.zeros(len(wide)), wide["L3 Following"], color="#4c78a8", edgecolors="#333333", linewidths=0.35, alpha=0.75, s=26, zorder=3)
        ax.scatter(np.ones(len(wide)), wide["L4 Following"], color="#f58518", edgecolors="#333333", linewidths=0.35, alpha=0.75, s=26, zorder=3)
        diff = means[0] - means[1]
        ax.set_title(f"{metric.label_en}\nDiff={diff:+.2f}", fontsize=10.5, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["L3", "L4"])
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

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(summary: pd.DataFrame, anova_tables: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append("4.3.3 控制权限对跟驰风格接受度的影响")
    lines.append("")
    lines.append("差值定义为 L3 Following - L4 Following；对紧张感而言，负值代表 L3 条件下紧张感更低。")
    lines.append("")

    aggressive = summary[summary["style_key"] == "aggressive"].copy()
    lines.append("Aggressive 风格的配对比较：")
    for metric_key in ["comfort", "smoothness", "expectation", "trust", "tension", "relaxation", "comfort_rank"]:
        row = aggressive[aggressive["metric_key"] == metric_key].iloc[0]
        direction = "高于" if row["mean_diff"] > 0 else "低于"
        if bool(row["lower_is_better"]):
            direction = "低于" if row["mean_diff"] < 0 else "高于"
        lines.append(
            f"- {row['metric']}: L3={row['l3_mean']:.2f}, L4={row['l4_mean']:.2f}, "
            f"差值={row['mean_diff']:+.2f}, t({int(row['n']) - 1})={row['t']:.2f}, "
            f"p={p_text(row['p_t'])}。"
        )

    lines.append("")
    lines.append("总体解释：")
    lines.append(
        "Aggressive 在预期一致性、信任度、舒适度、平稳性和放松感上均表现为 L3 高于 L4，"
        "紧张感则表现为 L3 低于 L4。该模式支持“感知控制权提高激进跟驰风格接受度”的解释。"
    )
    lines.append(
        "Conservative 在多数正向指标上表现为 L4 高于 L3，说明控制权限效应并非无差别提升所有风格，"
        "而是具有风格选择性。Neutral 与 Self 的 L3-L4 差异整体较小。"
    )
    lines.append("")

    if anova_tables:
        lines.append("2 x 4 重复测量 ANOVA 摘要：")
        for metric_key in ["expectation", "trust", "tension", "comfort", "comfort_rank"]:
            table = anova_tables.get(metric_key)
            if table is None:
                continue
            metric = metric_by_key(metric_key)
            lines.append(f"- {metric.label_cn}:")
            for _, row in table.iterrows():
                if {"Num DF", "Den DF", "F Value", "Pr > F"}.issubset(row.index):
                    lines.append(
                        f"  {row['effect']}: F({row['Num DF']:.0f}, {row['Den DF']:.0f})="
                        f"{row['F Value']:.2f}, p={p_text(row['Pr > F'])}"
                    )
                else:
                    if pd.isna(row["F"]) or not np.isfinite(float(row["F"])):
                        lines.append(f"  {row['effect']}: not estimable for this transformed metric.")
                    else:
                        lines.append(
                            f"  {row['effect']}: F({row['DF']:.0f}, {row['Error DF']:.0f})="
                            f"{row['F']:.2f}, p={p_text(row['p'])}, "
                            f"partial eta^2={row['partial_eta_sq']:.3f}"
                        )
    else:
        lines.append("未生成重复测量 ANOVA 表：当前环境可能未安装 statsmodels。")

    (OUTPUT_DIR / "4.3.3_H2_结果摘要.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = load_following_detail(df)
    summary = paired_summary(detail)
    desc = descriptive_table(detail)
    gain = control_gain_wide(summary)
    benefit_gain = benefit_gain_wide(summary)
    anova_tables = rm_anova_tables(detail)

    detail.to_csv(OUTPUT_DIR / "4.3.3_H2_following_detail.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "4.3.3_H2_L3_L4_following_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "4.3.3_H2_L3_minus_L4_paired_tests.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(OUTPUT_DIR / "4.3.3_H2_control_gain_table.csv", encoding="utf-8-sig")
    benefit_gain.to_csv(OUTPUT_DIR / "4.3.3_H2_benefit_coded_gain_table.csv", encoding="utf-8-sig")
    for metric_key, table in anova_tables.items():
        table.to_csv(OUTPUT_DIR / f"4.3.3_H2_rm_anova_{metric_key}.csv", index=False, encoding="utf-8-sig")

    plot_gain_heatmap(
        gain,
        OUTPUT_DIR / "4.3.3_H2_control_gain_heatmap.png",
        "Control Gain: L3 Following - L4 Following",
        "Raw differences are shown in cells. Positive values mean L3 > L4; for tension, negative values indicate lower tension in L3.",
    )
    plot_gain_heatmap(
        benefit_gain,
        OUTPUT_DIR / "4.3.3_H2_benefit_coded_gain_heatmap.png",
        "Benefit-Coded Control Gain",
        "For tension, the sign is reversed so positive values consistently indicate a more favorable L3 evaluation.",
    )
    plot_permission_style_interaction(detail, OUTPUT_DIR / "4.3.3_H2_permission_style_interaction.png")
    plot_following_means(detail, OUTPUT_DIR / "4.3.3_H2_L3_L4_following_mean_comparison.png")
    plot_aggressive_pairs(detail, OUTPUT_DIR / "4.3.3_H2_aggressive_paired_changes.png")
    write_summary(summary, anova_tables)

    print(f"detail rows: {len(detail)}")
    print(f"paired rows: {len(summary)}")
    print(f"anova tables: {len(anova_tables)}")
    print(f"output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

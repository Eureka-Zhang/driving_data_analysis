# -*- coding: utf-8 -*-
r"""
Section 4.3.2: H1 analysis for Self-style subjective advantages.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_h1_self_advantage.py
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
OUTPUT_DIR = BASE_DIR / "analysis_output_h1_self_advantage"

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
CONDITIONS = [
    ("l3", "L3 Following"),
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
    integer_only: bool = False


METRICS = [
    Metric("comfort", "Comfort", 11, 0, 100, [0, 20, 40, 60, 80, 100]),
    Metric("comfort_rank", "Comfort rank", 11, 1, 4, [1, 2, 3, 4]),
    Metric("expectation", "Expectation", 14, 1, 5, [1, 2, 3, 4, 5], True),
    Metric("trust", "Trust", 15, 1, 5, [1, 2, 3, 4, 5], True),
]


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def classify_condition(group_name: str) -> str | None:
    name = str(group_name).strip().lower()
    if name.startswith("l3"):
        return "l3"
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


def p_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "+"
    return ""


def paired_tests(a: pd.Series, b: pd.Series) -> dict:
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = pair["a"] - pair["b"]
    n = int(len(diff))
    if n < 2:
        return {"n": n, "mean_diff": np.nan, "sd_diff": np.nan, "t": np.nan, "p_t": np.nan, "dz": np.nan}

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    dz = mean_diff / sd_diff if sd_diff > 0 else np.nan
    try:
        from scipy.stats import ttest_rel

        result = ttest_rel(pair["a"], pair["b"], nan_policy="omit")
        t_stat, p_t = float(result.statistic), float(result.pvalue)
    except ImportError:
        t_stat, p_t = np.nan, np.nan

    return {
        "n": n,
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
        "p_t": round(p_t, 6) if pd.notna(p_t) else np.nan,
        "dz": round(dz, 4) if pd.notna(dz) else np.nan,
    }


def load_detail(df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
        }
    )
    detail["condition"] = detail["group"].map(classify_condition)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["subject", "group", "condition", "style"]).copy()

    for metric in METRICS:
        if metric.key == "comfort_rank":
            continue
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)

    detail["comfort_rank"] = detail.groupby(["subject", "condition"])["comfort"].rank(method="average")
    return detail


def metric_wide(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    return detail.pivot_table(index="subject", columns=["condition", "style"], values=metric_key, aggfunc="mean")


def descriptives(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for condition_key, condition_label in CONDITIONS:
            for style in STYLE_ORDER:
                values = detail.loc[
                    (detail["condition"] == condition_key) & (detail["style"] == style),
                    metric.key,
                ].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": condition_label,
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


def self_vs_others(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        wide = metric_wide(detail, metric.key)
        for condition_key, condition_label in CONDITIONS:
            self_col = (condition_key, "self")
            if self_col not in wide.columns:
                continue
            for style in STYLE_ORDER:
                if style == "self":
                    continue
                other_col = (condition_key, style)
                if other_col not in wide.columns:
                    continue
                test = paired_tests(wide[self_col], wide[other_col])
                rows.append(
                    {
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": condition_label,
                        "condition_key": condition_key,
                        "comparison": f"Self - {STYLE_LABELS[style]}",
                        "other_style": STYLE_LABELS[style],
                        "self_mean": round(float(wide[self_col].mean()), 3),
                        "other_mean": round(float(wide[other_col].mean()), 3),
                        **test,
                    }
                )
    return pd.DataFrame(rows)


def self_gap_vs_best_other(desc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        metric_desc = desc[desc["metric_key"] == metric.key]
        for condition_key, condition_label in CONDITIONS:
            sub = metric_desc[metric_desc["condition"] == condition_label]
            self_mean = float(sub.loc[sub["style"] == "Self", "mean"].iloc[0])
            others = sub[sub["style"] != "Self"].copy()
            best = others.loc[others["mean"].idxmax()]
            rows.append(
                {
                    "metric": metric.label,
                    "metric_key": metric.key,
                    "condition": condition_label,
                    "self_mean": round(self_mean, 3),
                    "best_other_style": best["style"],
                    "best_other_mean": round(float(best["mean"]), 3),
                    "self_minus_best_other": round(self_mean - float(best["mean"]), 3),
                }
            )
    return pd.DataFrame(rows)


def plot_expectation_means(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    metric = next(m for m in METRICS if m.key == "expectation")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Expectation Consistency by Style", fontsize=14, fontweight="bold", y=1.03)

    x = np.arange(len(STYLE_ORDER))
    for ax, (condition_key, condition_label) in zip(axes, CONDITIONS):
        means, ses = [], []
        for style in STYLE_ORDER:
            values = detail.loc[
                (detail["condition"] == condition_key) & (detail["style"] == style),
                metric.key,
            ].dropna()
            means.append(float(values.mean()))
            ses.append(float(values.std(ddof=1) / np.sqrt(len(values))))
        colors = [STYLE_COLORS[s] for s in STYLE_ORDER]
        ax.bar(x, means, yerr=ses, color=colors, edgecolor="#333333", linewidth=0.7, alpha=0.78, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE_LABELS[s] for s in STYLE_ORDER], rotation=18, ha="right")
        ax.set_ylim(0.6, 5.45)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(condition_label, fontsize=11, fontweight="bold")
        ax.axhline(3, color="#666666", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        for i, mean in enumerate(means):
            ax.text(i, 0.92, f"{mean:.2f}", ha="center", va="center", fontsize=8.5, color="#333333")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
    axes[0].set_ylabel("average score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_self_vs_others_heatmap(tests: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    sub = tests[tests["metric_key"] == "expectation"].copy()
    row_labels = [label for _, label in CONDITIONS]
    col_labels = ["Self - Aggressive", "Self - Conservative", "Self - Neutral"]
    values = np.full((len(row_labels), len(col_labels)), np.nan)
    annotations = [["" for _ in col_labels] for _ in row_labels]
    for i, condition in enumerate(row_labels):
        for j, comparison in enumerate(col_labels):
            row = sub[(sub["condition"] == condition) & (sub["comparison"] == comparison)]
            if row.empty:
                continue
            r0 = row.iloc[0]
            values[i, j] = float(r0["mean_diff"])
            annotations[i][j] = f"{values[i, j]:+.2f}{p_stars(float(r0['p_t']))}"

    max_abs = np.nanmax(np.abs(values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values, cmap="RdYlGn_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Self Advantage in Expectation Consistency", fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Self minus comparison style", fontsize=9)
    ax.text(
        0,
        -0.28,
        "Cell values are paired mean differences. + p<.10, * p<.05, ** p<.01, *** p<.001.",
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


def plot_self_gap_heatmap(gap: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    metric_order = ["Expectation", "Trust", "Comfort", "Comfort rank"]
    condition_labels = [label for _, label in CONDITIONS]
    values = gap.pivot(index="metric", columns="condition", values="self_minus_best_other").reindex(metric_order)[condition_labels]
    max_abs = np.nanmax(np.abs(values.to_numpy(dtype=float)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values.to_numpy(dtype=float), cmap="RdYlGn_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_xticklabels(condition_labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(metric_order)))
    ax.set_yticklabels(metric_order)
    ax.set_title("Self Gap Against Best Non-Self Style", fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Self mean - best non-Self mean", fontsize=9)
    ax.text(
        0,
        -0.28,
        "Positive values indicate Self is the highest-scoring style; negative values indicate another style is higher.",
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


def write_summary(desc: pd.DataFrame, tests: pd.DataFrame, gap: pd.DataFrame) -> None:
    lines = []
    lines.append("4.3.2 H1 Self-style advantage summary")
    lines.append("")
    lines.append("Recommended figures:")
    lines.append("- 4.3.2_H1_expectation_by_condition_style.png")
    lines.append("- 4.3.2_H1_self_vs_others_expectation_heatmap.png")
    lines.append("- 4.3.2_H1_self_gap_vs_best_other_heatmap.png")
    lines.append("")

    exp_gap = gap[gap["metric"] == "Expectation"]
    lines.append("Expectation consistency:")
    for _, row in exp_gap.iterrows():
        status = "highest" if row["self_minus_best_other"] > 0 else "not highest"
        lines.append(
            f"- {row['condition']}: Self={row['self_mean']:.2f}, best non-Self="
            f"{row['best_other_style']} ({row['best_other_mean']:.2f}), "
            f"gap={row['self_minus_best_other']:+.2f}; Self is {status}."
        )

    lines.append("")
    lines.append("Paired Self-vs-other expectation tests:")
    exp_tests = tests[tests["metric_key"] == "expectation"]
    for _, row in exp_tests.iterrows():
        lines.append(
            f"- {row['condition']}, {row['comparison']}: diff={row['mean_diff']:+.2f}, "
            f"t({int(row['n']) - 1})={row['t']:.2f}, p={row['p_t']:.3f}."
        )

    lines.append("")
    lines.append(
        "Interpretation: H1 is partially supported. Self has the highest expectation score in L3 Following and "
        "L4 Following, but not in L4 Overtaking. The strongest paired effects are Self > Conservative in L3 Following "
        "and Self > Aggressive in L4 Following."
    )
    (OUTPUT_DIR / "4.3.2_H1_result_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = load_detail(df)
    desc = descriptives(detail)
    tests = self_vs_others(detail)
    gap = self_gap_vs_best_other(desc)

    detail.to_csv(OUTPUT_DIR / "4.3.2_H1_detail.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "4.3.2_H1_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_DIR / "4.3.2_H1_self_vs_others_paired_tests.csv", index=False, encoding="utf-8-sig")
    gap.to_csv(OUTPUT_DIR / "4.3.2_H1_self_gap_vs_best_other.csv", index=False, encoding="utf-8-sig")

    plot_expectation_means(detail, OUTPUT_DIR / "4.3.2_H1_expectation_by_condition_style.png")
    plot_self_vs_others_heatmap(tests, OUTPUT_DIR / "4.3.2_H1_self_vs_others_expectation_heatmap.png")
    plot_self_gap_heatmap(gap, OUTPUT_DIR / "4.3.2_H1_self_gap_vs_best_other_heatmap.png")
    write_summary(desc, tests, gap)

    print(f"detail rows: {len(detail)}")
    print(f"test rows: {len(tests)}")
    print(f"output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

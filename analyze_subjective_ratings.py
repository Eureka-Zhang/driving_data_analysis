# -*- coding: utf-8 -*-
r"""
Generate descriptive statistics figures for section 4.3.1.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_subjective_ratings.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "356953624_按序号_自动驾驶系统乘坐体验问卷_241_240.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis_output_subjective"

GROUP_COL_INDEX = 10
SUBJECT_COL_INDEX = 9
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
SUPER_GROUPS = [
    ("l3", "L3 Following"),
    ("l4 follow", "L4 Following"),
    ("l4 overtake", "L4 Overtaking"),
]


@dataclass(frozen=True)
class Metric:
    key: str
    label_cn: str
    title_en: str
    column_index: int
    y_label: str
    y_min: float
    y_max: float
    neutral: float
    ticks: list[float]
    text_y: float
    integer_only: bool = False


METRICS = [
    Metric("comfort", "舒适度", "Comfort", 11, "comfort score", 0, 100, 50, [0, 20, 40, 60, 80, 100], 10),
    Metric("smoothness", "平稳性", "Smoothness", 12, "average score", 1, 5, 3, [1, 2, 3, 4, 5], 1.45, True),
    Metric("expectation", "预期一致性", "Expectation Consistency", 14, "average score", 1, 5, 3, [1, 2, 3, 4, 5], 1.45, True),
    Metric("trust", "信任度", "Trust", 15, "average score", 1, 5, 3, [1, 2, 3, 4, 5], 1.45, True),
    Metric("tension", "紧张感", "Tension", 16, "average score", 1, 5, 3, [1, 2, 3, 4, 5], 1.45, True),
    Metric("relaxation", "放松感", "Relaxation", 17, "average score", 1, 5, 3, [1, 2, 3, 4, 5], 1.45, True),
]

RANKED_COMFORT = Metric(
    "comfort_block_rank",
    "舒适度_区组内排序分",
    "Comfort Block Rank",
    11,
    "within-block comfort rank",
    1,
    4,
    2.5,
    [1, 2, 3, 4],
    1.25,
)


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def classify_super_group(group_name: str) -> str | None:
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


def metric_detail(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
    group = df.iloc[:, GROUP_COL_INDEX]
    detail = pd.DataFrame({"group": group, "score": score}).dropna(subset=["group", "score"])

    valid = detail["score"].between(metric.y_min, metric.y_max)
    if metric.integer_only:
        valid &= (detail["score"] % 1 == 0)
    detail = detail.loc[valid].copy()
    detail["super_group"] = detail["group"].map(classify_super_group)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["super_group", "style"])
    return detail


def ranked_comfort_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Rank comfort within each subject and block across the four driving styles."""
    score = pd.to_numeric(df.iloc[:, RANKED_COMFORT.column_index], errors="coerce")
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
            "raw_score": score,
        }
    ).dropna(subset=["subject", "group", "raw_score"])
    detail = detail.loc[detail["raw_score"].between(0, 100)].copy()
    detail["super_group"] = detail["group"].map(classify_super_group)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["super_group", "style"])

    detail["score"] = detail.groupby(["subject", "super_group"])["raw_score"].rank(method="average")
    return detail


def stats_for_metric(detail: pd.DataFrame, metric: Metric) -> list[dict]:
    rows = []
    for sg_key, sg_label in SUPER_GROUPS:
        sub = detail[detail["super_group"] == sg_key]
        for style in STYLE_ORDER:
            values = sub.loc[sub["style"] == style, "score"].astype(float).to_numpy()
            if len(values) == 0:
                continue
            rows.append(
                {
                    "metric": metric.label_cn,
                    "metric_en": metric.title_en,
                    "condition": sg_label,
                    "style": STYLE_LABELS[style],
                    "n": len(values),
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else np.nan,
                    "median": round(float(np.median(values)), 3),
                    "min": round(float(np.min(values)), 3),
                    "max": round(float(np.max(values)), 3),
                }
            )
    return rows


def plot_metric(detail: pd.DataFrame, metric: Metric, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{metric.title_en} Analysis", fontsize=13, fontweight="bold", y=1.03)

    rng = np.random.default_rng(2026)
    for ax, (sg_key, sg_title) in zip(axes, SUPER_GROUPS):
        sub = detail[detail["super_group"] == sg_key]
        data, labels, colors, styles = [], [], [], []
        for style in STYLE_ORDER:
            values = sub.loc[sub["style"] == style, "score"].astype(float).to_numpy()
            if len(values) == 0:
                continue
            data.append(values)
            labels.append(STYLE_LABELS[style])
            colors.append(STYLE_COLORS[style])
            styles.append(style)

        x_positions = np.arange(1, len(data) + 1)
        box = ax.boxplot(
            data,
            positions=x_positions,
            tick_labels=labels,
            patch_artist=True,
            widths=0.52,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 5.5,
            },
            medianprops={"color": "#333333", "linewidth": 1.5},
            whiskerprops={"color": "#333333", "linewidth": 1.0, "clip_on": True},
            capprops={"color": "#333333", "linewidth": 1.0, "clip_on": True},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 4.5,
                "alpha": 0.8,
            },
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.9)

        for x_pos, values, style in zip(x_positions, data, styles):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                s=32,
                color=STYLE_COLORS[style],
                edgecolors="#333333",
                linewidths=0.45,
                alpha=0.9,
                zorder=4,
                clip_on=True,
            )
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            ax.text(
                x_pos,
                metric.text_y,
                f"M={mean_value:.2f}\nMed={median_value:.1f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="#333333",
                clip_on=True,
            )

        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.tick_params(axis="x", pad=2)
        pad = 0.35 if metric.y_max <= 5 else 5
        ax.set_ylim(metric.y_min - pad, metric.y_max + pad)
        ax.set_yticks(metric.ticks)
        ax.set_title(sg_title, fontsize=11, fontweight="bold", color="black")
        if ax is axes[0]:
            ax.set_ylabel(metric.y_label)
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.axhline(metric.neutral, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=STYLE_COLORS[s], edgecolor="#333333", label=STYLE_LABELS[s])
        for s in STYLE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.045),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def subjective_heatmap_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [metric.title_en for metric in METRICS]
    condition_order = [label for _, label in SUPER_GROUPS]
    style_order = [STYLE_LABELS[style] for style in STYLE_ORDER]

    sub = stats_df[stats_df["metric_en"].isin(metric_order)].copy()
    table = sub.pivot_table(
        index=["condition", "style"],
        columns="metric_en",
        values="mean",
        aggfunc="mean",
    )
    row_index = pd.MultiIndex.from_product(
        [condition_order, style_order],
        names=["condition", "style"],
    )
    return table.reindex(index=row_index, columns=metric_order)


def plot_subjective_heatmap(table: pd.DataFrame, out_path: Path) -> bool:
    """Mean-score heatmap, using the H2 heatmap style and column-wise normalization."""
    if not HAS_MPL:
        return False
    setup_font()

    values = table.astype(float)
    normed = values.copy()
    for col in normed.columns:
        col_values = values[col].to_numpy(dtype=float)
        center = np.nanmean(col_values)
        max_abs = np.nanmax(np.abs(col_values - center))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
        normed[col] = (values[col] - center) / max_abs

    row_labels = [f"{condition} | {style}" for condition, style in values.index]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(normed.to_numpy(), cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Subjective Mean Scores by Condition and Style", fontsize=13, fontweight="bold", pad=12)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            text = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.8, color="#222222")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Column-centered normalized mean", fontsize=9)
    ax.text(
        0.0,
        -0.16,
        "Cell labels show raw means. Colors are normalized within each metric column, following the H2 heatmap palette.",
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


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    all_stats: list[dict] = []

    for idx, metric in enumerate(METRICS, start=1):
        detail = metric_detail(df, metric)
        detail.to_csv(OUTPUT_DIR / f"4.3.1_{idx}_{metric.label_cn}_原始评分.csv", index=False, encoding="utf-8-sig")
        all_stats.extend(stats_for_metric(detail, metric))
        out_path = OUTPUT_DIR / f"4.3.1_{idx}_{metric.label_cn}_分组对比.png"
        plotted = plot_metric(detail, metric, out_path)
        print(f"{idx}. {metric.label_cn}: n={len(detail)}, figure={'ok' if plotted else 'skipped'} -> {out_path}")

    ranked_detail = ranked_comfort_detail(df)
    ranked_detail.to_csv(
        OUTPUT_DIR / "4.3.1_1b_舒适度_区组内排序分_原始评分.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_stats.extend(stats_for_metric(ranked_detail, RANKED_COMFORT))
    ranked_out = OUTPUT_DIR / "4.3.1_1b_舒适度_区组内排序分_分组对比.png"
    plotted = plot_metric(ranked_detail, RANKED_COMFORT, ranked_out)
    print(
        f"1b. {RANKED_COMFORT.label_cn}: n={len(ranked_detail)}, "
        f"figure={'ok' if plotted else 'skipped'} -> {ranked_out}"
    )

    stats_df = pd.DataFrame(all_stats)
    stats_path = OUTPUT_DIR / "4.3.1_各条件下主观评分_描述统计.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    heatmap_table = subjective_heatmap_table(stats_df)
    heatmap_table.to_csv(OUTPUT_DIR / "4.3.1_主观评分均值热图矩阵.csv", encoding="utf-8-sig")
    heatmap_path = OUTPUT_DIR / "4.3.1_主观评分均值热图.png"
    heatmap_plotted = plot_subjective_heatmap(heatmap_table, heatmap_path)
    print(f"stats -> {stats_path}")
    print(f"heatmap={'ok' if heatmap_plotted else 'skipped'} -> {heatmap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

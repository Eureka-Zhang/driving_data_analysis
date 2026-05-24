# -*- coding: utf-8 -*-
r"""
Section 4.4: 3 x 4 repeated-measures ANOVA for subjective ratings.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_44_rm_anova.py
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
OUTPUT_DIR = BASE_DIR / "analysis_output_44_rm_anova"

SUBJECT_COL_INDEX = 9
GROUP_COL_INDEX = 10

CONDITION_ORDER = ["l3", "l4 follow", "l4 overtake"]
CONDITION_LABELS = {
    "l3": "L3 Following",
    "l4 follow": "L4 Following",
    "l4 overtake": "L4 Overtaking",
}
STYLE_ORDER = ["aggressive", "consecutive", "neutral", "self"]
STYLE_LABELS = {
    "aggressive": "Aggressive",
    "consecutive": "Conservative",
    "neutral": "Neutral",
    "self": "Self",
}


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
    Metric("expectation", "Expectation", 14, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("trust", "Trust", 15, 1, 5, [1, 2, 3, 4, 5], False, True),
    Metric("tension", "Tension", 16, 1, 5, [1, 2, 3, 4, 5], True, True),
    Metric("relaxation", "Relaxation", 17, 1, 5, [1, 2, 3, 4, 5], False, True),
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


def metric_by_key(metric_key: str) -> Metric:
    return next(m for m in METRICS if m.key == metric_key)


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<.001"
    return f"{p_value:.3f}"


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
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)
    return detail


def descriptives(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for condition in CONDITION_ORDER:
            for style in STYLE_ORDER:
                values = detail.loc[
                    (detail["condition"] == condition) & (detail["style"] == style),
                    metric.key,
                ].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": CONDITION_LABELS[condition],
                        "style": STYLE_LABELS[style],
                        "n": int(values.size),
                        "mean": round(float(values.mean()), 3),
                        "std": round(float(values.std(ddof=1)), 3),
                        "median": round(float(values.median()), 3),
                    }
                )
    return pd.DataFrame(rows)


def manual_rm_anova_3x4(detail: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import f as f_dist
    except ImportError:
        raise RuntimeError("scipy is required for ANOVA p-values")

    rows = []
    a, b = len(CONDITION_ORDER), len(STYLE_ORDER)
    for metric in METRICS:
        wide = detail.pivot_table(
            index="subject",
            columns=["condition", "style"],
            values=metric.key,
            aggfunc="mean",
        )
        cols = [(c, s) for c in CONDITION_ORDER for s in STYLE_ORDER]
        if not all(col in wide.columns for col in cols):
            continue
        wide = wide[cols].dropna()
        n = len(wide)
        if n < 2:
            continue

        y = wide.to_numpy(dtype=float).reshape(n, a, b)
        grand = y.mean()
        subj_mean = y.mean(axis=(1, 2))
        cond_mean = y.mean(axis=(0, 2))
        style_mean = y.mean(axis=(0, 1))
        cell_mean = y.mean(axis=0)
        subj_cond_mean = y.mean(axis=2)
        subj_style_mean = y.mean(axis=1)

        ss_cond = b * n * np.sum((cond_mean - grand) ** 2)
        ss_style = a * n * np.sum((style_mean - grand) ** 2)
        ss_inter = n * np.sum((cell_mean - cond_mean[:, None] - style_mean[None, :] + grand) ** 2)
        ss_sxcond = b * np.sum((subj_cond_mean - subj_mean[:, None] - cond_mean[None, :] + grand) ** 2)
        ss_sxstyle = a * np.sum((subj_style_mean - subj_mean[:, None] - style_mean[None, :] + grand) ** 2)
        ss_sxinter = np.sum(
            (
                y
                - subj_cond_mean[:, :, None]
                - subj_style_mean[:, None, :]
                - cell_mean[None, :, :]
                + subj_mean[:, None, None]
                + cond_mean[None, :, None]
                + style_mean[None, None, :]
                - grand
            )
            ** 2
        )

        specs = [
            ("Condition", ss_cond, a - 1, ss_sxcond, (n - 1) * (a - 1)),
            ("Style", ss_style, b - 1, ss_sxstyle, (n - 1) * (b - 1)),
            ("Condition:Style", ss_inter, (a - 1) * (b - 1), ss_sxinter, (n - 1) * (a - 1) * (b - 1)),
        ]
        for effect, ss, df_num, ss_err, df_den in specs:
            ms = ss / df_num
            ms_err = ss_err / df_den
            f_value = ms / ms_err if ms_err > 0 else np.nan
            p_value = float(f_dist.sf(f_value, df_num, df_den)) if pd.notna(f_value) else np.nan
            eta_p2 = ss / (ss + ss_err) if (ss + ss_err) > 0 else np.nan
            rows.append(
                {
                    "metric": metric.label,
                    "metric_key": metric.key,
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
                    "n_subjects": n,
                }
            )
    return pd.DataFrame(rows)


def simple_effect_pairs(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        wide = detail.pivot_table(index="subject", columns=["condition", "style"], values=metric.key, aggfunc="mean")
        for style in STYLE_ORDER:
            comparisons = [
                ("L3 Following - L4 Following", ("l3", style), ("l4 follow", style)),
                ("L4 Following - L4 Overtaking", ("l4 follow", style), ("l4 overtake", style)),
            ]
            for label, a_col, b_col in comparisons:
                if a_col not in wide.columns or b_col not in wide.columns:
                    continue
                pair = pd.concat([wide[a_col], wide[b_col]], axis=1, keys=["a", "b"]).dropna()
                diff = pair["a"] - pair["b"]
                n = len(diff)
                if n < 2:
                    continue
                try:
                    from scipy.stats import ttest_rel

                    test = ttest_rel(pair["a"], pair["b"], nan_policy="omit")
                    t_stat, p_val = float(test.statistic), float(test.pvalue)
                except ImportError:
                    t_stat, p_val = np.nan, np.nan
                rows.append(
                    {
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "style": STYLE_LABELS[style],
                        "comparison": label,
                        "n": n,
                        "mean_a": round(float(pair["a"].mean()), 4),
                        "mean_b": round(float(pair["b"].mean()), 4),
                        "mean_diff": round(float(diff.mean()), 4),
                        "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
                        "p": round(p_val, 6) if pd.notna(p_val) else np.nan,
                        "dz": round(float(diff.mean() / diff.std(ddof=1)), 4) if diff.std(ddof=1) > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_interactions(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Condition x Style Interaction", fontsize=14, fontweight="bold")

    x = np.arange(len(STYLE_ORDER))
    condition_specs = [
        ("l3", "L3 Following", "#4c78a8", "o"),
        ("l4 follow", "L4 Following", "#f58518", "s"),
        ("l4 overtake", "L4 Overtaking", "#54a24b", "D"),
    ]
    for ax, metric in zip(axes.flat, METRICS):
        for condition, label, color, marker in condition_specs:
            means, ses = [], []
            for style in STYLE_ORDER:
                values = detail.loc[(detail["condition"] == condition) & (detail["style"] == style), metric.key].dropna()
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

    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.06), frameon=True, framealpha=0.95, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_effect_summary(anova: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    effects = ["Condition", "Style", "Condition:Style"]
    metrics = [m.label for m in METRICS]
    values = anova.pivot(index="metric", columns="effect", values="partial_eta_sq").reindex(metrics)[effects]
    pvals = anova.pivot(index="metric", columns="effect", values="p").reindex(metrics)[effects]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(effects)))
    ax.set_xticklabels(effects, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title("ANOVA Effect Sizes (Partial Eta Squared)", fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            eta = values.iloc[i, j]
            p = pvals.iloc[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "+" if p < 0.1 else ""
            ax.text(j, i, f"{eta:.3f}{star}", ha="center", va="center", fontsize=9, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("partial eta squared", fontsize=9)
    ax.text(0.0, -0.25, "+ p<.10, * p<.05, ** p<.01, *** p<.001.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(anova: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = []
    lines.append("4.4 3 x 4 repeated-measures ANOVA summary")
    lines.append("")
    for effect in ["Style", "Condition", "Condition:Style"]:
        lines.append(effect)
        sub = anova[anova["effect"] == effect]
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['metric']}: F({int(row['DF'])}, {int(row['Error DF'])})={row['F']:.2f}, "
                f"p={p_text(row['p'])}, partial eta^2={row['partial_eta_sq']:.3f}."
            )
        lines.append("")

    lines.append("Planned contrasts to interpret interactions")
    for metric_key in ["expectation", "trust", "tension", "comfort", "relaxation"]:
        metric = metric_by_key(metric_key)
        lines.append(f"- {metric.label}:")
        sub = contrasts[contrasts["metric_key"] == metric_key]
        focus = sub[
            ((sub["comparison"] == "L3 Following - L4 Following") & (sub["style"] == "Aggressive"))
            | ((sub["comparison"] == "L4 Following - L4 Overtaking") & (sub["style"] == "Aggressive"))
            | ((sub["comparison"] == "L4 Following - L4 Overtaking") & (sub["style"] == "Self"))
        ]
        for _, row in focus.iterrows():
            lines.append(
                f"  {row['comparison']}, {row['style']}: diff={row['mean_diff']:+.2f}, "
                f"t({int(row['n']) - 1})={row['t']:.2f}, p={p_text(row['p'])}."
            )
    (OUTPUT_DIR / "4.4_rm_anova_result_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = load_detail(df)
    desc = descriptives(detail)
    anova = manual_rm_anova_3x4(detail)
    contrasts = simple_effect_pairs(detail)

    detail.to_csv(OUTPUT_DIR / "4.4_rm_anova_detail.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "4.4_rm_anova_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    anova.to_csv(OUTPUT_DIR / "4.4_rm_anova_3x4_results.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUTPUT_DIR / "4.4_planned_contrasts_H2_H3.csv", index=False, encoding="utf-8-sig")

    plot_interactions(detail, OUTPUT_DIR / "4.4_condition_style_interaction_plot.png")
    plot_effect_summary(anova, OUTPUT_DIR / "4.4_rm_anova_effect_size_heatmap.png")
    write_summary(anova, contrasts)

    print(f"detail rows: {len(detail)}")
    print(f"anova rows: {len(anova)}")
    print(f"contrast rows: {len(contrasts)}")
    print(f"output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

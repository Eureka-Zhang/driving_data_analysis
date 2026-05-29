# -*- coding: utf-8 -*-
r"""
Section 4.3.2: H1 analysis — individualized consistent style vs style mismatch.

Operationalization (longitudinal following tasks):
  - Aligned / individualized-consistent: Self vehicle OR preset matching Following Label
  - Mismatch: the two non-own presets (mean)
  - Extreme mismatch: Aggressive <-> Conservative opposite; Neutral uses mean(Aggressive, Conservative)

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
STYLE_FILE = BASE_DIR / "style.txt"
OUTPUT_DIR = BASE_DIR / "analysis_output_h1_self_advantage"

SUBJECT_COL_INDEX = 9
GROUP_COL_INDEX = 10

STYLE_ORDER = ["aggressive", "consecutive", "neutral", "self"]
PRESET_STYLES = ["aggressive", "consecutive", "neutral"]
STYLE_LABELS = {
    "aggressive": "Aggressive",
    "consecutive": "Conservative",
    "neutral": "Neutral",
    "self": "Self",
}
LABEL_TO_STYLE_KEY = {
    "Aggressive": "aggressive",
    "Conservative": "consecutive",
    "Neutral": "neutral",
}
STYLE_COLORS = {
    "aggressive": "#d62728",
    "consecutive": "#2ca02c",
    "neutral": "#1f77b4",
    "self": "#ff7f0e",
    "aligned": "#9467bd",
    "mismatch": "#8c564b",
    "extreme": "#bcbd22",
}
H1_CONDITIONS = [
    ("l3", "L3 Following"),
    ("l4 follow", "L4 Following"),
]
ALL_CONDITIONS = [
    *H1_CONDITIONS,
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
    higher_is_better: bool = True


METRICS = [
    Metric("comfort", "Comfort", 11, 0, 100, [0, 20, 40, 60, 80, 100]),
    Metric("comfort_rank", "Comfort rank", 11, 1, 4, [1, 2, 3, 4], higher_is_better=False),
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


def score_for_comparison(value: float, metric: Metric) -> float:
    if pd.isna(value):
        return np.nan
    if metric.higher_is_better:
        return float(value)
    return -float(value)


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


def load_style_labels() -> pd.DataFrame:
    if not STYLE_FILE.exists():
        raise FileNotFoundError(f"style.txt not found: {STYLE_FILE}")

    labels = pd.read_csv(STYLE_FILE, sep="\t")
    labels = labels.rename(columns={"Driver": "subject"}).copy()
    labels["subject"] = labels["subject"].astype(str).str.strip()
    labels["Following Label"] = labels["Following Label"].astype(str).str.strip()
    return labels[["subject", "Following Label"]]


def load_detail(df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX].astype(str).str.strip(),
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


def attach_following_labels(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = detail.merge(labels, on="subject", how="inner")
    merged["own_style_key"] = merged["Following Label"].map(LABEL_TO_STYLE_KEY)
    return merged.dropna(subset=["own_style_key"]).copy()


def metric_wide(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    return detail.pivot_table(index="subject", columns=["condition", "style"], values=metric_key, aggfunc="mean")


def mismatch_preset_styles(own_style_key: str) -> list[str]:
    return [style for style in PRESET_STYLES if style != own_style_key]


def extreme_mismatch_styles(own_style_key: str) -> list[str]:
    if own_style_key == "aggressive":
        return ["consecutive"]
    if own_style_key == "consecutive":
        return ["aggressive"]
    return ["aggressive", "consecutive"]


def cell_value(wide: pd.DataFrame, subject: str, condition_key: str, style_key: str, metric: Metric) -> float:
    col = (condition_key, style_key)
    if col not in wide.columns or subject not in wide.index:
        return np.nan
    return score_for_comparison(wide.loc[subject, col], metric)


def mean_values(values: list[float]) -> float:
    clean = [v for v in values if pd.notna(v)]
    if not clean:
        return np.nan
    return float(np.mean(clean))


def build_subject_contrasts(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labeled = attach_following_labels(detail, labels)
    subjects = sorted(labeled["subject"].unique())
    label_map = labels.set_index("subject")["Following Label"].to_dict()
    own_map = {subject: LABEL_TO_STYLE_KEY[label_map[subject]] for subject in subjects if subject in label_map}

    rows = []
    for metric in METRICS:
        wide = metric_wide(labeled, metric.key)
        for condition_key, condition_label in H1_CONDITIONS:
            for subject in subjects:
                if subject not in wide.index:
                    continue
                own_key = own_map.get(subject)
                if not own_key:
                    continue
                self_score = cell_value(wide, subject, condition_key, "self", metric)
                own_score = cell_value(wide, subject, condition_key, own_key, metric)
                mismatch_styles = mismatch_preset_styles(own_key)
                mismatch_scores = [
                    cell_value(wide, subject, condition_key, style_key, metric) for style_key in mismatch_styles
                ]
                extreme_styles = extreme_mismatch_styles(own_key)
                extreme_scores = [
                    cell_value(wide, subject, condition_key, style_key, metric) for style_key in extreme_styles
                ]

                aligned_scores = [v for v in [self_score, own_score] if pd.notna(v)]
                rows.append(
                    {
                        "subject": subject,
                        "following_label": label_map.get(subject),
                        "own_style_key": own_key,
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": condition_label,
                        "condition_key": condition_key,
                        "self_score": self_score,
                        "own_label_score": own_score,
                        "aligned_best": max(aligned_scores) if aligned_scores else np.nan,
                        "aligned_mean": mean_values(aligned_scores),
                        "mismatch_mean": mean_values(mismatch_scores),
                        "extreme_mismatch": mean_values(extreme_scores),
                        "mismatch_styles": "/".join(STYLE_LABELS[s] for s in mismatch_styles),
                        "extreme_styles": "/".join(STYLE_LABELS[s] for s in extreme_styles),
                    }
                )
    return pd.DataFrame(rows)


def contrast_tests(
    contrasts: pd.DataFrame,
    label_filter: str | None = None,
) -> pd.DataFrame:
    rows = []
    data = contrasts if label_filter is None else contrasts[contrasts["following_label"] == label_filter]

    comparisons = [
        ("aligned_best", "mismatch_mean", "Aligned best (Self/own) - Mismatch mean"),
        ("aligned_mean", "mismatch_mean", "Aligned mean (Self+own)/2 - Mismatch mean"),
        ("self_score", "mismatch_mean", "Self - Mismatch mean"),
        ("own_label_score", "mismatch_mean", "Own-label preset - Mismatch mean"),
        ("self_score", "own_label_score", "Self - Own-label preset"),
        ("aligned_best", "extreme_mismatch", "Aligned best - Extreme mismatch"),
        ("self_score", "extreme_mismatch", "Self - Extreme mismatch"),
    ]

    for metric in METRICS:
        sub_metric = data[data["metric_key"] == metric.key]
        for condition_key, condition_label in H1_CONDITIONS:
            sub = sub_metric[sub_metric["condition_key"] == condition_key]
            if sub.empty:
                continue
            for left_col, right_col, comparison in comparisons:
                test = paired_tests(sub[left_col], sub[right_col])
                rows.append(
                    {
                        "analysis_scope": label_filter or "All subjects",
                        "following_label": label_filter or "All",
                        "metric": metric.label,
                        "metric_key": metric.key,
                        "condition": condition_label,
                        "condition_key": condition_key,
                        "comparison": comparison,
                        "left_column": left_col,
                        "right_column": right_col,
                        "left_mean": round(float(sub[left_col].mean()), 4),
                        "right_mean": round(float(sub[right_col].mean()), 4),
                        **test,
                    }
                )
    return pd.DataFrame(rows)


def descriptives(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for condition_key, condition_label in ALL_CONDITIONS:
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
        for condition_key, condition_label in ALL_CONDITIONS:
            self_col = (condition_key, "self")
            if self_col not in wide.columns:
                continue
            for style in STYLE_ORDER:
                if style == "self":
                    continue
                other_col = (condition_key, style)
                if other_col not in wide.columns:
                    continue
                left = wide[self_col].map(lambda v: score_for_comparison(v, metric))
                right = wide[other_col].map(lambda v: score_for_comparison(v, metric))
                test = paired_tests(left, right)
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


def plot_following_expectation_by_style(detail: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    metric = next(m for m in METRICS if m.key == "expectation")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("H1 Following Tasks: Expectation by Vehicle Style", fontsize=14, fontweight="bold", y=1.03)

    x = np.arange(len(STYLE_ORDER))
    for ax, (condition_key, condition_label) in zip(axes, H1_CONDITIONS):
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


def plot_aligned_vs_mismatch_heatmap(tests: pd.DataFrame, out_path: Path, metric_key: str, title: str) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    sub = tests[
        (tests["analysis_scope"] == "All subjects")
        & (tests["metric_key"] == metric_key)
        & tests["comparison"].isin(
            [
                "Aligned best (Self/own) - Mismatch mean",
                "Self - Mismatch mean",
                "Own-label preset - Mismatch mean",
                "Aligned best - Extreme mismatch",
            ]
        )
    ].copy()
    if sub.empty:
        return False

    row_labels = [label for _, label in H1_CONDITIONS]
    col_labels = [
        "Aligned best - Mismatch mean",
        "Self - Mismatch mean",
        "Own-label preset - Mismatch mean",
        "Aligned best - Extreme mismatch",
    ]
    short_cols = ["Aligned - Mismatch", "Self - Mismatch", "Own label - Mismatch", "Aligned - Extreme"]
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

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(len(short_cols)))
    ax.set_xticklabels(short_cols, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("paired mean difference (higher = left condition better)", fontsize=9)
    ax.text(
        0,
        -0.30,
        "Aligned = max(Self, own-label preset). Mismatch = mean of two non-own presets. + p<.10, * p<.05, ** p<.01.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
    )
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_by_label_expectation(contrasts: pd.DataFrame, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    sub = contrasts[contrasts["metric_key"] == "expectation"].copy()
    if sub.empty:
        return False

    label_groups = ["Aggressive", "Conservative", "Neutral"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("H1 Expectation: Aligned vs Mismatch by Following Label Group", fontsize=13, fontweight="bold", y=1.03)

    categories = ["Self", "Own label", "Mismatch mean", "Extreme mismatch"]
    x = np.arange(len(categories))
    width = 0.22
    palette = ["#ff7f0e", "#9467bd", "#8c564b", "#bcbd22"]

    for ax, (condition_key, condition_label) in zip(axes, H1_CONDITIONS):
        for idx, label in enumerate(label_groups):
            part = sub[(sub["condition_key"] == condition_key) & (sub["following_label"] == label)]
            if part.empty:
                continue
            means = [
                float(part["self_score"].mean()),
                float(part["own_label_score"].mean()),
                float(part["mismatch_mean"].mean()),
                float(part["extreme_mismatch"].mean()),
            ]
            offset = (idx - 1) * width
            ax.bar(x + offset, means, width=width, label=f"{label} (n={part['subject'].nunique()})", color=palette, alpha=0.55 + 0.15 * idx, edgecolor="#333333", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=12, ha="right")
        ax.set_ylim(0.6, 5.2)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(condition_label, fontsize=11, fontweight="bold")
        ax.axhline(3, color="#666666", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("average score")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(contrasts: pd.DataFrame, tests: pd.DataFrame) -> None:
    lines = []
    lines.append("4.3.2 H1 individualized consistent style vs mismatch")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Scope: longitudinal following tasks only (L3 Following, L4 Following).")
    lines.append("Labels: Following Label from style.txt.")
    lines.append("")
    lines.append("Operationalization:")
    lines.append("  Aligned best = max(Self, own-label preset) per subject.")
    lines.append("  Aligned mean = mean(Self, own-label preset).")
    lines.append("  Mismatch mean = mean of the two non-own presets.")
    lines.append("  Extreme mismatch = opposite preset (Aggressive<->Conservative); Neutral uses mean(Aggressive, Conservative).")
    lines.append("")
    lines.append("Recommended figures:")
    lines.append("- 4.3.2_H1_following_expectation_by_style.png")
    lines.append("- 4.3.2_H1_aligned_vs_mismatch_expectation_heatmap.png")
    lines.append("- 4.3.2_H1_aligned_vs_mismatch_trust_heatmap.png")
    lines.append("- 4.3.2_H1_expectation_by_label_group.png")
    lines.append("")

    primary = tests[(tests["analysis_scope"] == "All subjects") & (tests["metric_key"] == "expectation")]
    lines.append("【Primary H1 tests — Expectation】")
    for _, row in primary.iterrows():
        lines.append(
            f"- {row['condition']}, {row['comparison']}: diff={row['mean_diff']:+.3f}, "
            f"t({int(row['n']) - 1})={row['t']:.2f}, p={row['p_t']:.3f}, dz={row['dz']:.2f}."
        )

    lines.append("")
    lines.append("【By Following Label group — Expectation, Aligned best vs Mismatch】")
    for label in ["Aggressive", "Conservative", "Neutral"]:
        sub = tests[
            (tests["following_label"] == label)
            & (tests["metric_key"] == "expectation")
            & (tests["comparison"] == "Aligned best (Self/own) - Mismatch mean")
        ]
        for _, row in sub.iterrows():
            lines.append(
                f"- {label}, {row['condition']}: diff={row['mean_diff']:+.3f}, "
                f"n={int(row['n'])}, p={row['p_t']:.3f}."
            )

    lines.append("")
    lines.append("【Subject-level aligned-best advantage rate — Expectation】")
    exp = contrasts[contrasts["metric_key"] == "expectation"].copy()
    for condition_key, condition_label in H1_CONDITIONS:
        sub = exp[exp["condition_key"] == condition_key]
        better = (sub["aligned_best"] > sub["mismatch_mean"]).mean()
        lines.append(
            f"- {condition_label}: {100 * better:.1f}% subjects with aligned_best > mismatch_mean "
            f"(mean gap={float((sub['aligned_best'] - sub['mismatch_mean']).mean()):+.3f})."
        )

    lines.append("")
    lines.append(
        "Interpretation note: Positive differences indicate the individualized-consistent side "
        "(Self and/or own-label preset) outperforms mismatched presets on the comparison-transformed scale."
    )
    (OUTPUT_DIR / "4.3.2_H1_result_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = load_detail(df)
    labels = load_style_labels()

    contrasts = build_subject_contrasts(detail, labels)
    tests_all = contrast_tests(contrasts)
    subgroup_tests = []
    for label in sorted(contrasts["following_label"].dropna().unique()):
        subgroup_tests.append(contrast_tests(contrasts, label_filter=label))
    tests = pd.concat([tests_all, *subgroup_tests], ignore_index=True)

    desc = descriptives(detail)
    legacy_tests = self_vs_others(detail)

    contrasts.to_csv(OUTPUT_DIR / "4.3.2_H1_subject_contrasts.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_DIR / "4.3.2_H1_aligned_vs_mismatch_tests.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "4.3.2_H1_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    legacy_tests.to_csv(OUTPUT_DIR / "4.3.2_H1_self_vs_others_supplementary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUTPUT_DIR / "4.3.2_H1_detail.csv", index=False, encoding="utf-8-sig")

    plot_following_expectation_by_style(detail, OUTPUT_DIR / "4.3.2_H1_following_expectation_by_style.png")
    plot_aligned_vs_mismatch_heatmap(
        tests,
        OUTPUT_DIR / "4.3.2_H1_aligned_vs_mismatch_expectation_heatmap.png",
        "expectation",
        "H1 Expectation: Aligned vs Mismatch (Following Tasks)",
    )
    plot_aligned_vs_mismatch_heatmap(
        tests,
        OUTPUT_DIR / "4.3.2_H1_aligned_vs_mismatch_trust_heatmap.png",
        "trust",
        "H1 Trust: Aligned vs Mismatch (Following Tasks)",
    )
    plot_by_label_expectation(contrasts, OUTPUT_DIR / "4.3.2_H1_expectation_by_label_group.png")
    write_summary(contrasts, tests)

    print(f"detail rows: {len(detail)}")
    print(f"contrast rows: {len(contrasts)}")
    print(f"test rows: {len(tests)}")
    print(f"output: {OUTPUT_DIR}")
    print()
    print(tests_all[tests_all["metric_key"] == "expectation"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

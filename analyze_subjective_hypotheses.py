# -*- coding: utf-8 -*-
r"""
Analyze subjective-rating hypotheses for sections 4.3.2-4.3.4.

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_subjective_hypotheses.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "356953624_按序号_自动驾驶系统乘坐体验问卷_241_240.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis_output_hypotheses"

SUBJECT_COL_INDEX = 9
GROUP_COL_INDEX = 10

STYLE_ORDER = ["aggressive", "consecutive", "neutral", "self"]
STYLE_LABELS = {
    "aggressive": "Aggressive",
    "consecutive": "Conservative",
    "neutral": "Neutral",
    "self": "Self",
}

# Change these display aliases if Alpha/Beta/Gamma have a different mapping
# in the thesis text.
STYLE_ALIASES = {
    "aggressive": "Alpha",
    "consecutive": "Beta",
    "neutral": "Gamma",
    "self": "Self",
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
    column_index: int
    y_min: float
    y_max: float
    integer_only: bool = False


METRICS = [
    Metric("comfort", "舒适度", 11, 0, 100),
    Metric("comfort_rank", "舒适度_区组内排序分", 11, 1, 4),
    Metric("smoothness", "平稳性", 12, 1, 5, True),
    Metric("expectation", "预期一致性", 14, 1, 5, True),
    Metric("trust", "信任度", 15, 1, 5, True),
    Metric("tension", "紧张感", 16, 1, 5, True),
    Metric("relaxation", "放松感", 17, 1, 5, True),
]

KEY_METRICS = ["expectation", "trust", "comfort", "comfort_rank"]


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


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<.001"
    return f"{p_value:.3f}"


def paired_tests(a: pd.Series, b: pd.Series) -> dict:
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = pair["a"] - pair["b"]
    n = int(len(diff))
    if n < 2:
        return {"n": n, "mean_diff": np.nan, "sd_diff": np.nan, "t": np.nan, "p_t": np.nan, "w": np.nan, "p_w": np.nan, "dz": np.nan}

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    dz = mean_diff / sd_diff if sd_diff > 0 else np.nan

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

    return {
        "n": n,
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "t": round(t_stat, 4) if pd.notna(t_stat) else np.nan,
        "p_t": round(p_t, 6) if pd.notna(p_t) else np.nan,
        "w": round(w_stat, 4) if pd.notna(w_stat) else np.nan,
        "p_w": round(p_w, 6) if pd.notna(p_w) else np.nan,
        "dz": round(dz, 4) if pd.notna(dz) else np.nan,
    }


def base_detail(df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
        }
    )
    detail["super_group"] = detail["group"].map(classify_super_group)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["subject", "group", "super_group", "style"]).copy()

    for metric in METRICS:
        if metric.key == "comfort_rank":
            continue
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)

    detail["comfort_rank"] = detail.groupby(["subject", "super_group"])["comfort"].rank(method="average")
    return detail


def metric_wide(detail: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    return detail.pivot_table(
        index="subject",
        columns=["super_group", "style"],
        values=metric_key,
        aggfunc="mean",
    )


def descriptives(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for sg_key, sg_label in SUPER_GROUPS:
            for style in STYLE_ORDER:
                values = detail.loc[
                    (detail["super_group"] == sg_key) & (detail["style"] == style),
                    metric.key,
                ].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "metric": metric.label_cn,
                        "metric_key": metric.key,
                        "condition": sg_label,
                        "style": STYLE_LABELS[style],
                        "alias": STYLE_ALIASES[style],
                        "n": int(values.size),
                        "mean": round(float(values.mean()), 3),
                        "std": round(float(values.std(ddof=1)), 3),
                        "median": round(float(values.median()), 3),
                        "min": round(float(values.min()), 3),
                        "max": round(float(values.max()), 3),
                    }
                )
    return pd.DataFrame(rows)


def analyze_h1(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_key in KEY_METRICS:
        metric_label = next(m.label_cn for m in METRICS if m.key == metric_key)
        wide = metric_wide(detail, metric_key)
        for sg_key, sg_label in SUPER_GROUPS:
            means = {}
            for style in STYLE_ORDER:
                col = (sg_key, style)
                if col in wide:
                    means[style] = float(wide[col].mean())
            top_style = max(means, key=means.get) if means else None
            for style in STYLE_ORDER:
                if style == "self":
                    continue
                self_col, other_col = (sg_key, "self"), (sg_key, style)
                if self_col not in wide or other_col not in wide:
                    continue
                test = paired_tests(wide[self_col], wide[other_col])
                rows.append(
                    {
                        "section": "4.3.2",
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "condition": sg_label,
                        "comparison": f"Self - {STYLE_ALIASES[style]}",
                        "self_mean": round(float(wide[self_col].mean()), 3),
                        "other_mean": round(float(wide[other_col].mean()), 3),
                        "top_style_by_mean": STYLE_ALIASES.get(top_style, top_style),
                        **test,
                    }
                )
    return pd.DataFrame(rows)


def analyze_h2(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_key in KEY_METRICS:
        metric_label = next(m.label_cn for m in METRICS if m.key == metric_key)
        wide = metric_wide(detail, metric_key)
        for style in STYLE_ORDER:
            l3_col, l4_col = ("l3", style), ("l4 follow", style)
            if l3_col not in wide or l4_col not in wide:
                continue
            test = paired_tests(wide[l3_col], wide[l4_col])
            rows.append(
                {
                    "section": "4.3.3",
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "comparison": f"L3 Following {STYLE_ALIASES[style]} - L4 Following {STYLE_ALIASES[style]}",
                    "style": STYLE_LABELS[style],
                    "alias": STYLE_ALIASES[style],
                    "l3_mean": round(float(wide[l3_col].mean()), 3),
                    "l4_follow_mean": round(float(wide[l4_col].mean()), 3),
                    **test,
                }
            )
    return pd.DataFrame(rows)


def analyze_h3_pairs(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_key in KEY_METRICS:
        metric_label = next(m.label_cn for m in METRICS if m.key == metric_key)
        wide = metric_wide(detail, metric_key)
        for style in STYLE_ORDER:
            follow_col, overtake_col = ("l4 follow", style), ("l4 overtake", style)
            if follow_col not in wide or overtake_col not in wide:
                continue
            test = paired_tests(wide[follow_col], wide[overtake_col])
            rows.append(
                {
                    "section": "4.3.4",
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "comparison": f"L4 Following {STYLE_ALIASES[style]} - L4 Overtaking {STYLE_ALIASES[style]}",
                    "style": STYLE_LABELS[style],
                    "alias": STYLE_ALIASES[style],
                    "l4_follow_mean": round(float(wide[follow_col].mean()), 3),
                    "l4_overtake_mean": round(float(wide[overtake_col].mean()), 3),
                    **test,
                }
            )
    return pd.DataFrame(rows)


def winner_by_subject(detail: pd.DataFrame, metric_key: str, super_group: str) -> pd.Series:
    sub = detail[detail["super_group"] == super_group]
    wide = sub.pivot_table(index="subject", columns="style", values=metric_key, aggfunc="mean")
    wide = wide[[s for s in STYLE_ORDER if s in wide.columns]].dropna(how="all")

    winners = []
    for subject, row in wide.iterrows():
        max_value = row.max()
        tied = [STYLE_ALIASES[s] for s in STYLE_ORDER if s in row.index and pd.notna(row[s]) and row[s] == max_value]
        winners.append((subject, "/".join(tied)))
    return pd.Series(dict(winners), name=super_group)


def analyze_h3_migration(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrices = {}
    for metric_key in KEY_METRICS:
        follow = winner_by_subject(detail, metric_key, "l4 follow")
        overtake = winner_by_subject(detail, metric_key, "l4 overtake")
        both = pd.concat([follow, overtake], axis=1, keys=["L4 Following", "L4 Overtaking"]).dropna()
        matrix = pd.crosstab(both["L4 Following"], both["L4 Overtaking"])
        matrices[metric_key] = matrix
    return matrices


def write_summary(h1: pd.DataFrame, h2: pd.DataFrame, h3: pd.DataFrame, matrices: dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("4.3.2 Self 风格的预期一致性优势")
    exp_h1 = h1[h1["metric_key"] == "expectation"].copy()
    for _, row in exp_h1.iterrows():
        direction = "高于" if row["mean_diff"] > 0 else "低于"
        lines.append(
            f"- {row['condition']}: Self 均值 {row['self_mean']:.2f}，"
            f"{row['comparison'].replace('Self - ', '对比 ')} 均值 {row['other_mean']:.2f}，"
            f"Self {direction}对方 {abs(row['mean_diff']):.2f}，配对 t 检验 p={p_text(row['p_t'])}。"
        )
    lines.append("")
    lines.append("4.3.3 不同控制权限下的风格接受差异")
    gamma = h2[(h2["metric_key"] == "expectation") & (h2["alias"] == "Gamma")]
    for _, row in gamma.iterrows():
        direction = "更高" if row["mean_diff"] > 0 else "更低"
        lines.append(
            f"- Gamma 预期一致性: L3 跟驰均值 {row['l3_mean']:.2f}，"
            f"L4 跟驰均值 {row['l4_follow_mean']:.2f}，L3 {direction} {abs(row['mean_diff']):.2f}，"
            f"p={p_text(row['p_t'])}。"
        )
    lines.append("")
    lines.append("4.3.4 跟驰 L4 与超车 L4 的偏好差异")
    exp_h3 = h3[h3["metric_key"] == "expectation"]
    for _, row in exp_h3.iterrows():
        direction = "高于" if row["mean_diff"] > 0 else "低于"
        lines.append(
            f"- {row['alias']}: L4 跟驰均值 {row['l4_follow_mean']:.2f}，"
            f"L4 超车均值 {row['l4_overtake_mean']:.2f}，跟驰 {direction}超车 {abs(row['mean_diff']):.2f}，"
            f"p={p_text(row['p_t'])}。"
        )
    lines.append("")
    lines.append("偏好迁移矩阵已按各指标输出；单元格为被试人数，行=L4 跟驰最偏好风格，列=L4 超车最偏好风格。")

    (OUTPUT_DIR / "4.3.2-4.3.4_结果摘要.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXCEL_FILE.exists():
        print(f"Excel file not found: {EXCEL_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_FILE)
    detail = base_detail(df)
    detail.to_csv(OUTPUT_DIR / "subjective_hypotheses_detail.csv", index=False, encoding="utf-8-sig")

    desc = descriptives(detail)
    h1 = analyze_h1(detail)
    h2 = analyze_h2(detail)
    h3 = analyze_h3_pairs(detail)
    matrices = analyze_h3_migration(detail)

    desc.to_csv(OUTPUT_DIR / "4.3.2-4.3.4_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    h1.to_csv(OUTPUT_DIR / "4.3.2_H1_self_vs_other_paired_tests.csv", index=False, encoding="utf-8-sig")
    h2.to_csv(OUTPUT_DIR / "4.3.3_H2_l3_vs_l4_follow_paired_tests.csv", index=False, encoding="utf-8-sig")
    h3.to_csv(OUTPUT_DIR / "4.3.4_H3_l4_follow_vs_overtake_paired_tests.csv", index=False, encoding="utf-8-sig")
    for metric_key, matrix in matrices.items():
        matrix.to_csv(OUTPUT_DIR / f"4.3.4_preference_migration_{metric_key}.csv", encoding="utf-8-sig")

    write_summary(h1, h2, h3, matrices)

    print(f"detail rows: {len(detail)}")
    print(f"H1 rows: {len(h1)}")
    print(f"H2 rows: {len(h2)}")
    print(f"H3 rows: {len(h3)}")
    print(f"output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

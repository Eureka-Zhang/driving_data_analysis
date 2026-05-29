# -*- coding: utf-8 -*-
"""
分析「被试前测问卷1.xlsx」的 T/U/V 列。

T/U/V 分别对应 Q1/Q2/Q3：
  - 生成 Q1/Q2/Q3 的箱线图 + 原始散点
  - 生成每个题目的扇形比例图
  - 输出频数表、扩展统计和文字报告

使用 carla 环境运行：
  C:\\Users\\16638\\miniconda3\\envs\\carla\\python.exe analyze_pretest_tuv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis_output_pretest_tuv"

COLUMN_CONFIG = {
    "Q1": {"index": 19, "excel_col": "T", "color": "#2ca02c"},
    "Q2": {"index": 20, "excel_col": "U", "color": "#1f77b4"},
    "Q3": {"index": 21, "excel_col": "V", "color": "#ff7f0e"},
}

LIKERT_MIN = 1
LIKERT_MAX = 5
TEST_MU = 3.0

SCORE_LABELS = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
}


def find_pretest_file() -> Path:
    candidates = sorted(BASE_DIR.glob("*.xlsx"))
    preferred = [
        p
        for p in candidates
        if "前测" in p.name and "问卷" in p.name and p.name.endswith("1.xlsx")
    ]
    if preferred:
        return preferred[0]

    fallback = [p for p in candidates if "问卷1" in p.name or p.name.endswith("1.xlsx")]
    if fallback:
        return fallback[0]

    raise FileNotFoundError("未找到类似「被试前测问卷1.xlsx」的 Excel 文件。")


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def clean_likert(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(raw, errors="coerce")
    valid_mask = values.notna() & (values % 1 == 0) & (values >= LIKERT_MIN) & (values <= LIKERT_MAX)
    valid = values.loc[valid_mask].astype(int)
    invalid = raw.loc[~valid_mask]
    return valid, invalid


def ci95(values: pd.Series) -> tuple[float, float]:
    arr = values.astype(float).to_numpy()
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean
    # n=20 时使用常见 t 临界值 2.093；其他样本量用正态近似，避免依赖 scipy。
    t_critical = 2.093 if n == 20 else 1.96
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean - t_critical * sem, mean + t_critical * sem


def summary_row(label: str, question: str, values: pd.Series) -> dict:
    lo, hi = ci95(values)
    return {
        "题号": label,
        "Excel列": COLUMN_CONFIG[label]["excel_col"],
        "题目": question,
        "样本量": len(values),
        "均值": round(float(values.mean()), 4),
        "标准差": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
        "中位数": round(float(values.median()), 4),
        "95%CI下限": round(lo, 4),
        "95%CI上限": round(hi, 4),
        "高于3分比例(%)": round(100 * (values > TEST_MU).sum() / len(values), 2),
        "大于等于4分比例(%)": round(100 * (values >= 4).sum() / len(values), 2),
    }


def frequency_table(label: str, values: pd.Series) -> pd.DataFrame:
    counts = values.value_counts().sort_index()
    rows = []
    for score in range(LIKERT_MIN, LIKERT_MAX + 1):
        n = int(counts.get(score, 0))
        rows.append(
            {
                "题号": label,
                "分值": score,
                "人数": n,
                "占比(%)": round(100 * n / len(values), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_boxplot(results: dict[str, dict], out_path: Path) -> None:
    setup_plot_style()

    labels = list(results.keys())
    data = [results[label]["valid"].astype(float).to_numpy() for label in labels]
    colors = [COLUMN_CONFIG[label]["color"] for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Pre-test Questionnaire", fontsize=14, fontweight="bold", y=1.02)

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.48,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 6,
        },
        medianprops={"color": "#333333", "linewidth": 1.5},
        whiskerprops={"color": "#333333", "linewidth": 1.0, "clip_on": True},
        capprops={"color": "#333333", "linewidth": 1.0, "clip_on": True},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 5,
            "alpha": 0.8,
        },
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.9)

    rng = np.random.default_rng(2026)
    for x_pos, label, values, color in zip(range(1, len(labels) + 1), labels, data, colors):
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), x_pos) + jitter,
            values,
            s=34,
            color=color,
            edgecolors="#333333",
            linewidths=0.45,
            alpha=0.85,
            zorder=3,
            clip_on=True,
        )
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.text(
            x_pos,
            1.45,
            f"M={mean_value:.2f}\nMed={median_value:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
            clip_on=True,
        )

    ax.set_ylim(LIKERT_MIN - 0.35, LIKERT_MAX + 0.45)
    ax.set_yticks(range(LIKERT_MIN, LIKERT_MAX + 1))
    ax.set_ylabel("average score")
    ax.axhline(TEST_MU, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
    ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pies(results: dict[str, dict], output_dir: Path) -> None:
    setup_plot_style()

    for label, result in results.items():
        freq = result["freq"].copy()
        freq = freq[freq["人数"] > 0]
        values = freq["人数"].tolist()
        labels = [f"{int(score)}分\n{pct:.1f}%" for score, pct in zip(freq["分值"], freq["占比(%)"])]
        colors = ["#d62728", "#ff7f0e", "#bbbbbb", "#1f77b4", "#2ca02c"]
        pie_colors = [colors[int(score) - 1] for score in freq["分值"]]

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        fig.patch.set_facecolor("white")
        ax.pie(
            values,
            labels=labels,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 10},
        )
        ax.set_title(f"{label} Score Proportion", fontsize=12, fontweight="bold")
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_扇形比例图.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Pre-test Score Proportions", fontsize=14, fontweight="bold", y=1.03)
    for ax, (label, result) in zip(axes, results.items()):
        freq = result["freq"].copy()
        freq = freq[freq["人数"] > 0]
        values = freq["人数"].tolist()
        labels = [f"{int(score)}分\n{pct:.1f}%" for score, pct in zip(freq["分值"], freq["占比(%)"])]
        colors = ["#d62728", "#ff7f0e", "#bbbbbb", "#1f77b4", "#2ca02c"]
        pie_colors = [colors[int(score) - 1] for score in freq["分值"]]
        ax.pie(
            values,
            labels=labels,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 9},
        )
        ax.set_title(label, fontweight="bold")
        ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "Q1_Q2_Q3_扇形比例图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(excel_file: Path, results: dict[str, dict], summary: pd.DataFrame, freq_all: pd.DataFrame) -> None:
    lines = [
        "被试前测问卷 T/U/V 列分析",
        "=" * 60,
        f"数据文件: {excel_file.name}",
        "",
        "【题目对应】",
    ]
    for label, result in results.items():
        lines.append(f"{label} ({COLUMN_CONFIG[label]['excel_col']}列): {result['question']}")

    lines += [
        "",
        "【描述统计】",
        summary.to_string(index=False),
        "",
        "【频数分布】",
        freq_all.to_string(index=False),
        "",
        "【输出文件】",
        f"箱型图: {OUTPUT_DIR / 'Q1_Q2_Q3_箱型图.png'}",
        f"合并扇形图: {OUTPUT_DIR / 'Q1_Q2_Q3_扇形比例图.png'}",
        f"统计表: {OUTPUT_DIR / 'Q1_Q2_Q3_扩展统计.csv'}",
        f"频数表: {OUTPUT_DIR / 'Q1_Q2_Q3_频数表.csv'}",
    ]
    (OUTPUT_DIR / "Q1_Q2_Q3_分析报告.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excel_file = find_pretest_file()
    df = pd.read_excel(excel_file)

    results: dict[str, dict] = {}
    summary_rows = []
    freq_tables = []

    for label, cfg in COLUMN_CONFIG.items():
        idx = cfg["index"]
        if df.shape[1] <= idx:
            raise ValueError(f"表格只有 {df.shape[1]} 列，不存在 {cfg['excel_col']} 列。")

        question = str(df.columns[idx])
        raw = df.iloc[:, idx]
        valid, invalid = clean_likert(raw)
        if valid.empty:
            raise ValueError(f"{label} / {cfg['excel_col']} 列没有有效的 1-5 分数据。")

        freq = frequency_table(label, valid)
        results[label] = {
            "question": question,
            "raw": raw,
            "valid": valid,
            "invalid": invalid,
            "freq": freq,
        }
        summary_rows.append(summary_row(label, question, valid))
        freq_tables.append(freq)

    summary = pd.DataFrame(summary_rows)
    freq_all = pd.concat(freq_tables, ignore_index=True)

    summary.to_csv(OUTPUT_DIR / "Q1_Q2_Q3_扩展统计.csv", index=False, encoding="utf-8-sig")
    freq_all.to_csv(OUTPUT_DIR / "Q1_Q2_Q3_频数表.csv", index=False, encoding="utf-8-sig")
    plot_boxplot(results, OUTPUT_DIR / "Q1_Q2_Q3_箱型图.png")
    plot_pies(results, OUTPUT_DIR)
    write_report(excel_file, results, summary, freq_all)

    print(f"分析完成: {excel_file.name}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

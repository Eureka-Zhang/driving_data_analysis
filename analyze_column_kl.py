# -*- coding: utf-8 -*-
"""
分析 L3 驾驶偏好调研 Excel 的 K、L 列（李克特 1–5 分）。

K: 知道脚下有踏板可以随时接管，让我感到更安心
L: 我认为我可以通过接管影响车辆行为

使用 carla 环境运行：
  C:\\Users\\16638\\miniconda3\\envs\\carla\\python.exe analyze_column_kl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analyze_column_n import (
    HAS_MPL,
    LIKERT_MAX,
    LIKERT_MIN,
    SCORE_LABELS,
    TAB10_BLUE,
    TAB10_GREEN,
    TAB10_ORANGE,
    TEST_MU,
    _apply_reference_axes_style,
    _format_extended_block,
    _setup_chinese_font,
    clean_scores,
    descriptive_stats,
    extended_stats_row,
    extended_stats_table,
    frequency_table,
)

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "361427626_按序号_L3驾驶偏好调研_20_20.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis_output_kl"

COLUMNS = {
    "K": {
        "index": 10,
        "short": "K: Pedal reassurance",
        "title_en": "Pedal Reassurance",
        "box_label": "Q1",
    },
    "L": {
        "index": 11,
        "short": "L: Takeover efficacy",
        "title_en": "Takeover Efficacy",
        "box_label": "Q2",
    },
}

Y_LABEL_PAD = 0.32


def load_excel() -> pd.DataFrame:
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"未找到文件: {EXCEL_FILE}")
    return pd.read_excel(EXCEL_FILE)


def load_column(df: pd.DataFrame, letter: str) -> tuple[pd.Series, str]:
    cfg = COLUMNS[letter]
    idx = cfg["index"]
    if df.shape[1] <= idx:
        raise ValueError(f"表格仅有 {df.shape[1]} 列，不存在第 {letter} 列")
    col_name = str(df.columns[idx])
    raw = pd.to_numeric(df.iloc[:, idx], errors="coerce")
    return raw, col_name


def _compute_ylim(means: list[float], stds: list[float]) -> float:
    tops = [m + s + Y_LABEL_PAD for m, s in zip(means, stds)]
    y_top = max(tops) if tops else float(LIKERT_MAX) + Y_LABEL_PAD
    return float(np.ceil(y_top * 4) / 4)


def plot_distribution(
    scores: pd.Series,
    col_name: str,
    title_en: str,
    out_path: Path,
    color: str = TAB10_BLUE,
) -> bool:
    if not HAS_MPL:
        return False
    _setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    bins = np.arange(LIKERT_MIN - 0.5, LIKERT_MAX + 1.5, 1)
    axes[0].hist(scores, bins=bins, edgecolor="white", color=color, alpha=0.85)
    axes[0].set_xticks(range(LIKERT_MIN, LIKERT_MAX + 1))
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution (histogram)")
    _apply_reference_axes_style(axes[0])

    freq = scores.value_counts().sort_index()
    x = list(range(LIKERT_MIN, LIKERT_MAX + 1))
    y = [int(freq.get(i, 0)) for i in x]
    axes[1].bar(x, y, color=color, edgecolor="#333333", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Count")
    for xi, yi in zip(x, y):
        if yi > 0:
            axes[1].text(xi, yi + 0.08, str(yi), ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Count by score")
    _apply_reference_axes_style(axes[1])

    short = col_name if len(col_name) <= 36 else col_name[:33] + "..."
    fig.suptitle(f"{title_en}\n{short}  (n={len(scores)})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_kl_comparison(
    results: dict[str, dict],
    out_path: Path,
) -> bool:
    """K / L 均值与标准差并排对比（仿 Naturalness Analysis 风格）。"""
    if not HAS_MPL:
        return False
    _setup_chinese_font()

    letters = ["K", "L"]
    means, stds, labels, colors = [], [], [], []
    color_map = {"K": TAB10_GREEN, "L": TAB10_BLUE}
    for letter in letters:
        ext = results[letter]["overall_ext"]
        means.append(float(ext["均值"]))
        stds.append(float(ext["标准差"]) if pd.notna(ext["标准差"]) else 0.0)
        labels.append(COLUMNS[letter]["title_en"])
        colors.append(color_map[letter])

    y_top = _compute_ylim(means, stds)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("L3 Takeover Perception: K vs L", fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        ecolor="#555555",
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "clip_on": True},
        zorder=3,
        clip_on=True,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(LIKERT_MIN, y_top)
    ax.set_ylabel("average score")
    _apply_reference_axes_style(ax)

    for bar, m, s in zip(bars, means, stds):
        label_y = min(m + s + 0.06, y_top - 0.06)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
            clip_on=True,
        )

    legend_handles = [
        Patch(facecolor=TAB10_GREEN, edgecolor="#333333", label="K: Pedal reassurance"),
        Patch(facecolor=TAB10_BLUE, edgecolor="#333333", label="L: Takeover efficacy"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_kl_scatter(valid_k: pd.Series, valid_l: pd.Series, out_path: Path) -> bool:
    """K–L 配对散点图（每位被试）。"""
    if not HAS_MPL or len(valid_k) != len(valid_l):
        return False
    _setup_chinese_font()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("white")
    ax.scatter(valid_k.values, valid_l.values, c=TAB10_ORANGE, s=80, edgecolors="#333333", linewidths=0.6, zorder=3)
    ax.plot([LIKERT_MIN, LIKERT_MAX], [LIKERT_MIN, LIKERT_MAX], "--", color="#999999", linewidth=1, label="y = x")
    ax.axhline(TEST_MU, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(TEST_MU, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(LIKERT_MIN - 0.3, LIKERT_MAX + 0.3)
    ax.set_ylim(LIKERT_MIN - 0.3, LIKERT_MAX + 0.3)
    ax.set_xlabel("K: Pedal reassurance")
    ax.set_ylabel("L: Takeover efficacy")
    ax.set_title("K vs L (paired subjects)", fontweight="bold")
    _apply_reference_axes_style(ax)

    if len(valid_k) >= 3:
        r = float(np.corrcoef(valid_k.values, valid_l.values)[0, 1])
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_freq_side_by_side(
    freq_k: pd.DataFrame,
    freq_l: pd.DataFrame,
    out_path: Path,
) -> bool:
    """K / L 频数分布并排柱状图。"""
    if not HAS_MPL:
        return False
    _setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Score Distribution: K vs L", fontsize=12, fontweight="bold")

    for ax, freq, letter, color in zip(
        axes,
        [freq_k, freq_l],
        ["K", "L"],
        [TAB10_GREEN, TAB10_BLUE],
    ):
        x = freq["分值"].tolist()
        y = freq["人数"].tolist()
        ax.bar(x, y, color=color, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(range(LIKERT_MIN, LIKERT_MAX + 1))
        ax.set_xlabel("Score")
        ax.set_title(COLUMNS[letter]["title_en"], fontweight="bold", color="black")
        for xi, yi in zip(x, y):
            if yi > 0:
                ax.text(xi, yi + 0.15, str(int(yi)), ha="center", va="bottom", fontsize=9)
        _apply_reference_axes_style(ax)

    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_kl_boxplot(results: dict[str, dict], out_path: Path) -> bool:
    """在同一张图中绘制 K/L 两个量表的箱型图，分别标注为 Q1/Q2。"""
    if not HAS_MPL:
        return False
    _setup_chinese_font()

    data = [
        results["K"]["valid"].astype(float).values,
        results["L"]["valid"].astype(float).values,
    ]
    labels = [COLUMNS["K"]["box_label"], COLUMNS["L"]["box_label"]]
    colors = [TAB10_GREEN, TAB10_BLUE]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("L3 Takeover Perception", fontsize=13, fontweight="bold", y=1.02)

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
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

    # 叠加轻微抖动的原始点，方便查看 20 名被试的离散情况。
    rng = np.random.default_rng(2026)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        x = rng.normal(i, 0.035, size=len(values))
        ax.scatter(
            x,
            values,
            s=28,
            color=color,
            edgecolors="#333333",
            linewidths=0.45,
            alpha=0.9,
            zorder=4,
            clip_on=True,
        )
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.text(
            i,
            1.45,
            f"M={mean_value:.2f}\nMed={median_value:.1f}",
            ha="center",
            va="center",
            fontsize=8.2,
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
    return True


def _format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"={p_value:.3f}"


def _format_ci(row: dict) -> str:
    return f"[{row['95%CI下限']:.2f}, {row['95%CI上限']:.2f}]"


def build_paper_style_paragraph(results: dict[str, dict], corr: float | None) -> str:
    """生成可直接放入论文结果部分的文字描述。"""
    q1 = results["K"]["overall_ext"]
    q2 = results["L"]["overall_ext"]
    q1_name = results["K"]["col_name"]
    q2_name = results["L"]["col_name"]

    parts = [
        "为进一步考察 L3 条件下被试对接管权限的主观感知，本文结合被试在驾驶偏好调研中填写的两个量表题项进行了分析。",
        (
            f"其中，Q1（{q1_name}）的评分均值为 {q1['均值']:.2f}/5，"
            f"标准差为 {q1['标准差']:.2f}，95% 置信区间为 {_format_ci(q1)}，"
            f"{q1['高于3分比例(%)']:.1f}% 的评分高于量表中性值 3，"
            f"{q1['大于等于4分比例(%)']:.1f}% 的评分达到 4 分及以上。"
            f"单样本 t 检验与 Wilcoxon 符号秩检验结果显示，Q1 评分显著高于中性水平"
            f"（t={q1['t检验统计量']:.2f}, p{_format_p_value(q1['t检验p值'])}；"
            f"Wilcoxon={q1['Wilcoxon统计量']:.0f}, p{_format_p_value(q1['Wilcoxon p值'])}），"
            "说明被试普遍认为踏板接管权限提升了其乘坐安心感。"
        ),
        (
            f"Q2（{q2_name}）的评分均值为 {q2['均值']:.2f}/5，"
            f"标准差为 {q2['标准差']:.2f}，95% 置信区间为 {_format_ci(q2)}，"
            f"{q2['高于3分比例(%)']:.1f}% 的评分高于中性值，"
            f"{q2['大于等于4分比例(%)']:.1f}% 的评分达到 4 分及以上。"
            f"单样本 t 检验与 Wilcoxon 符号秩检验同样表明，Q2 评分显著高于中性水平"
            f"（t={q2['t检验统计量']:.2f}, p{_format_p_value(q2['t检验p值'])}；"
            f"Wilcoxon={q2['Wilcoxon统计量']:.0f}, p{_format_p_value(q2['Wilcoxon p值'])}），"
            "说明被试整体认为自己能够通过接管对车辆行为产生影响。"
        ),
    ]
    if corr is not None:
        parts.append(f"此外，Q1 与 Q2 的配对评分相关系数为 r={corr:.2f}，可作为两类接管感知之间关系的补充参考。")
    return "\n".join(parts)


def write_column_report(
    letter: str,
    col_name: str,
    raw: pd.Series,
    valid: pd.Series,
    invalid: pd.Series,
    stats: pd.Series,
    freq: pd.DataFrame,
    overall_ext: dict,
    out_txt: Path,
) -> None:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"L3 驾驶偏好调研 — {letter} 列数据分析报告")
    lines.append("=" * 60)
    lines.append(f"数据文件: {EXCEL_FILE.name}")
    lines.append(f"题目: {col_name}")
    lines.append(f"量表范围: {LIKERT_MIN}–{LIKERT_MAX} 分（李克特）")
    lines.append("")
    lines.append("【数据清洗】")
    lines.append(f"  原始非空记录数: {raw.notna().sum()}")
    lines.append(f"  有效答卷数: {len(valid)}")
    lines.append(f"  剔除记录数: {len(invalid)}")
    if len(invalid):
        lines.append("  剔除的值:")
        for idx, val in invalid.items():
            lines.append(f"    行索引 {idx}: {val}")
    lines.append("")
    lines.append("【必备指标 — 总体】")
    lines.extend(_format_extended_block(overall_ext)[:2])
    lines.append("")
    lines.append("【推荐指标 — 总体】")
    lines.extend(_format_extended_block(overall_ext)[2:])
    lines.append("")
    lines.append("【补充描述统计】")
    lines.append(f"  中位数: {stats['50%']:.1f}")
    lines.append(f"  最小值: {int(stats['min'])}")
    lines.append(f"  最大值: {int(stats['max'])}")
    lines.append("")
    lines.append("【频数分布】")
    lines.append(freq.to_string(index=False))
    lines.append("")
    lines.append("【说明】")
    lines.append(f"  单样本 t / Wilcoxon 检验以 μ={TEST_MU} 为参照。")
    lines.append("")
    lines.append("=" * 60)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def write_combined_report(
    results: dict[str, dict],
    corr: float | None,
    out_txt: Path,
) -> None:
    paragraph = build_paper_style_paragraph(results, corr)
    lines = [
        "=" * 60,
        "L3 驾驶偏好调研 — K / L 列联合分析报告",
        "=" * 60,
        f"数据文件: {EXCEL_FILE.name}",
        "",
        "【论文式结果描述】",
        paragraph,
        "",
    ]
    for letter in ["K", "L"]:
        ext = results[letter]["overall_ext"]
        lines.append(f"【{letter} 列】{results[letter]['col_name']}")
        lines.extend(_format_extended_block(ext))
        lines.append("")
    if corr is not None:
        lines.append(f"【K–L 相关】Pearson r = {corr:.4f}  (n={len(results['K']['valid'])})")
    lines.append("")
    lines.append("=" * 60)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def analyze_column(df: pd.DataFrame, letter: str) -> dict:
    raw, col_name = load_column(df, letter)
    valid, invalid = clean_scores(raw)
    if valid.empty:
        raise ValueError(f"{letter} 列无有效 1–5 分数据")

    stats = descriptive_stats(valid)
    freq = frequency_table(valid)
    overall_ext = extended_stats_row(valid, group_name="总体")

    prefix = OUTPUT_DIR / f"{letter}列"
    write_column_report(
        letter,
        col_name,
        raw,
        valid,
        invalid,
        stats,
        freq,
        overall_ext,
        Path(f"{prefix}_分析报告.txt"),
    )
    freq.to_csv(f"{prefix}_频数表.csv", index=False, encoding="utf-8-sig")
    extended_stats_table([overall_ext]).to_csv(
        f"{prefix}_扩展统计.csv", index=False, encoding="utf-8-sig"
    )
    plot_distribution(
        valid,
        col_name,
        COLUMNS[letter]["title_en"],
        Path(f"{prefix}_分布图.png"),
        color=TAB10_GREEN if letter == "K" else TAB10_BLUE,
    )

    return {
        "letter": letter,
        "col_name": col_name,
        "raw": raw,
        "valid": valid,
        "invalid": invalid,
        "stats": stats,
        "freq": freq,
        "overall_ext": overall_ext,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_excel()

    results: dict[str, dict] = {}
    for letter in ["K", "L"]:
        results[letter] = analyze_column(df, letter)
        print(f"\n=== {letter} 列: {results[letter]['col_name']} ===")
        print(f"  n={len(results[letter]['valid'])}  均值={results[letter]['overall_ext']['均值']}")

    ext_rows = [results["K"]["overall_ext"], results["L"]["overall_ext"]]
    extended_stats_table(ext_rows).to_csv(
        OUTPUT_DIR / "K_L_扩展统计.csv", index=False, encoding="utf-8-sig"
    )

    vk, vl = results["K"]["valid"], results["L"]["valid"]
    common_idx = vk.index.intersection(vl.index)
    vk_a, vl_a = vk.loc[common_idx], vl.loc[common_idx]
    corr = float(np.corrcoef(vk_a.values, vl_a.values)[0, 1]) if len(common_idx) >= 3 else None

    write_combined_report(results, corr, OUTPUT_DIR / "K_L_联合分析报告.txt")
    (OUTPUT_DIR / "K_L_论文式结果描述.txt").write_text(
        build_paper_style_paragraph(results, corr),
        encoding="utf-8",
    )

    if HAS_MPL:
        plot_kl_comparison(results, OUTPUT_DIR / "K_L_均值对比.png")
        plot_kl_boxplot(results, OUTPUT_DIR / "K_L_箱型图.png")
        plot_freq_side_by_side(
            results["K"]["freq"],
            results["L"]["freq"],
            OUTPUT_DIR / "K_L_频数对比.png",
        )
        plot_kl_scatter(vk_a, vl_a, OUTPUT_DIR / "K_L_散点图.png")
        print("\n图表已保存至 analysis_output_kl/")
    else:
        print("\n未安装 matplotlib，跳过图表生成。")

    print(f"\n分析完成。输出目录: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

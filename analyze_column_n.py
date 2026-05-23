# -*- coding: utf-8 -*-
"""
分析问卷 Excel 第 N 列数据：
  「6、系统的驾驶风格显得自然，非常接近真实的人类驾驶员」

使用 carla 环境运行（在项目目录下）：
  C:\\Users\\16638\\miniconda3\\envs\\carla\\python.exe analyze_column_n.py

或在 PowerShell 中：
  & C:\\Users\\16638\\miniconda3\\envs\\carla\\python.exe .\\analyze_column_n.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "356953624_按序号_自动驾驶系统乘坐体验问卷_241_240.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis_output_n"
ROUND_CSV = OUTPUT_DIR / "N列_按实验轮次.csv"
EXTENDED_STATS_CSV = OUTPUT_DIR / "N列_扩展统计.csv"
ROUND_FIG = OUTPUT_DIR / "N列_按实验轮次_分组对比.png"
ROUND_SCORE_CSV = OUTPUT_DIR / "N列_按实验轮次_原始评分.csv"
ROUND_SCATTER_FIG = OUTPUT_DIR / "N列_按实验轮次_散点图.png"
COLUMN_INDEX = 13  # Excel 第 N 列（0-based 索引 13）
LIKERT_MIN, LIKERT_MAX = 1, 5
# 单样本检验参照值（量表中立点）
TEST_MU = 3.0
DRIVING_STYLES = ["aggressive", "consecutive", "neutral", "self"]
STYLE_LABELS = {
    "aggressive": "Aggressive",
    "consecutive": "Conservative",
    "neutral": "Neutral",
    "self": "Self",
}
# matplotlib tab10，与 Following Style 散点图一致
TAB10_RED = "#d62728"      # Aggressive
TAB10_BLUE = "#1f77b4"     # Neutral
TAB10_GREEN = "#2ca02c"    # Conservative / Consecutive
TAB10_ORANGE = "#ff7f0e"   # Self（第 4 类补充色）

STYLE_COLORS = {
    "aggressive": TAB10_RED,
    "consecutive": TAB10_GREEN,
    "neutral": TAB10_BLUE,
    "self": TAB10_ORANGE,
}
SUPER_GROUP_COLORS = {
    "l3": TAB10_GREEN,
    "l4 follow": TAB10_BLUE,
    "l4 overtake": TAB10_RED,
}
SUPER_GROUPS = [
    ("l3", "L3 Following"),
    ("l4 follow", "L4 Following"),
    ("l4 overtake", "L4 Overtaking"),
]
Y_LABEL_PAD = 0.32  # 柱顶数值标注留白
SCORE_LABELS = {
    1: "非常不同意",
    2: "不同意",
    3: "中立",
    4: "同意",
    5: "非常同意",
}


def load_column_n() -> tuple[pd.Series, str, pd.DataFrame]:
    """读取 Excel 并返回 N 列原始序列、列名及完整 DataFrame。"""
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"未找到文件: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)
    if df.shape[1] <= COLUMN_INDEX:
        raise ValueError(f"表格仅有 {df.shape[1]} 列，不存在第 N 列（索引 {COLUMN_INDEX}）")
    col_name = str(df.columns[COLUMN_INDEX])
    raw = pd.to_numeric(df.iloc[:, COLUMN_INDEX], errors="coerce")
    return raw, col_name, df


def clean_scores(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    清洗评分：保留 1–5 的整数分；剔除汇总行、非数值、越界值。
    返回 (有效分数, 被剔除记录)。
    """
    valid_mask = raw.notna() & (raw % 1 == 0) & (raw >= LIKERT_MIN) & (raw <= LIKERT_MAX)
    valid = raw.loc[valid_mask].astype(int)
    invalid = raw.loc[~valid_mask]
    return valid, invalid


def descriptive_stats(scores: pd.Series) -> pd.Series:
    return scores.describe()


def _t_critical_975(df: int) -> float:
    """双侧 95% 的 t 分位数；优先 scipy，否则查表/近似。"""
    if df < 1:
        return 1.96
    try:
        from scipy.stats import t as t_dist

        return float(t_dist.ppf(0.975, df))
    except ImportError:
        if df >= 60:
            return 1.96
        # 常用自由度查表（双侧 0.05）
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
            8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
            15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060,
            30: 2.042, 40: 2.021, 50: 2.009, 59: 2.000,
        }
        if df in table:
            return table[df]
        keys = sorted(k for k in table if k <= df)
        return table[keys[-1]] if keys else 1.96


def _t_pvalue_two_sided(t_stat: float, df: int) -> float:
    try:
        from scipy.stats import t as t_dist

        return float(2 * t_dist.sf(abs(t_stat), df))
    except ImportError:
        import math

        z = abs(t_stat)
        return float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _ci_95_mean(arr: np.ndarray) -> tuple[float, float]:
    """均值 95% 置信区间（t 分布）。"""
    n = len(arr)
    m = float(np.mean(arr))
    if n < 2:
        return m, m
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    h = _t_critical_975(n - 1) * sem
    return m - h, m + h


def _one_sample_ttest(arr: np.ndarray, mu0: float) -> tuple[float, float]:
    diff = arr.astype(float) - mu0
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1))
    if sd_d < 1e-12:
        return (0.0, 1.0) if abs(mean_d) < 1e-12 else (np.inf, 0.0)
    t_stat = mean_d / (sd_d / np.sqrt(n))
    p_val = _t_pvalue_two_sided(t_stat, n - 1)
    return t_stat, p_val


def _wilcoxon_signed_rank(arr: np.ndarray, mu0: float) -> tuple[float, float]:
    """Wilcoxon 符号秩检验（相对 mu0）；无 scipy 时用正态近似。"""
    diff = arr.astype(float) - mu0
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    if n == 0:
        return 0.0, 1.0
    if n == 1:
        return 1.0, 1.0

    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(diff, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        ranks = _rankdata(np.abs(diff))
        w_plus = float(ranks[diff > 0].sum())
        expected = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        if var <= 0:
            return w_plus, 1.0
        import math

        z = (w_plus - expected) / math.sqrt(var)
        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return w_plus, float(min(1.0, max(0.0, p_val)))


def _one_sample_tests(arr: np.ndarray, mu0: float = TEST_MU) -> dict[str, float]:
    """单样本 t 检验与 Wilcoxon 符号秩检验（相对 mu0）。"""
    out = {
        "t_stat": np.nan,
        "t_p": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
    }
    if len(arr) < 2:
        return out
    t_stat, t_p = _one_sample_ttest(arr, mu0)
    w_stat, w_p = _wilcoxon_signed_rank(arr, mu0)
    out["t_stat"] = t_stat
    out["t_p"] = t_p
    out["wilcoxon_stat"] = w_stat
    out["wilcoxon_p"] = w_p
    return out


def extended_stats_row(scores: pd.Series, group_name: str = "总体", mu0: float = TEST_MU) -> dict:
    """
    汇总必备 + 推荐指标：均值、标准差、95%CI、比例、单样本 t / Wilcoxon。
    """
    arr = scores.astype(float).values
    n = len(arr)
    mean = float(np.mean(arr)) if n else np.nan
    std = float(np.std(arr, ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan)
    ci_lo, ci_hi = _ci_95_mean(arr) if n else (np.nan, np.nan)
    tests = _one_sample_tests(arr, mu0) if n else _one_sample_tests(np.array([]), mu0)

    return {
        "组别": group_name,
        "样本量": n,
        "均值": round(mean, 4) if n else np.nan,
        "标准差": round(std, 4) if n and not np.isnan(std) else np.nan,
        "95%CI下限": round(ci_lo, 4) if n else np.nan,
        "95%CI上限": round(ci_hi, 4) if n else np.nan,
        "高于3分比例(%)": round(100 * (arr > 3).sum() / n, 2) if n else np.nan,
        "大于等于4分比例(%)": round(100 * (arr >= 4).sum() / n, 2) if n else np.nan,
        "t检验统计量": round(tests["t_stat"], 4) if not np.isnan(tests["t_stat"]) else np.nan,
        "t检验p值": round(tests["t_p"], 4) if not np.isnan(tests["t_p"]) else np.nan,
        "Wilcoxon统计量": round(tests["wilcoxon_stat"], 4)
        if not np.isnan(tests["wilcoxon_stat"])
        else np.nan,
        "Wilcoxon p值": round(tests["wilcoxon_p"], 4) if not np.isnan(tests["wilcoxon_p"]) else np.nan,
        "检验参照值": mu0,
    }


def extended_stats_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def frequency_table(scores: pd.Series) -> pd.DataFrame:
    counts = scores.value_counts().sort_index()
    total = len(scores)
    rows = []
    for score in range(LIKERT_MIN, LIKERT_MAX + 1):
        n = int(counts.get(score, 0))
        rows.append(
            {
                "分值": score,
                "含义": SCORE_LABELS.get(score, ""),
                "人数": n,
                "占比(%)": round(100 * n / total, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def group_stats(df: pd.DataFrame, scores: pd.Series, group_col_keyword: str) -> pd.DataFrame | None:
    """按「实验轮次」分组，输出均值/标准差/95%CI/比例/单样本检验等指标。"""
    candidates = [c for c in df.columns if group_col_keyword in str(c)]
    if not candidates:
        return None
    group_col = candidates[0]
    aligned = df.loc[scores.index, group_col]
    tmp = pd.DataFrame({"score": scores.values, "group": aligned.values}, index=scores.index)
    tmp = tmp.dropna(subset=["group"])
    if tmp.empty:
        return None

    rows = []
    for group_name, grp in tmp.groupby("group", sort=True):
        rows.append(extended_stats_row(grp["score"], group_name=str(group_name)))
    return extended_stats_table(rows)


def plot_distribution(scores: pd.Series, col_name: str, out_path: Path) -> bool:
    if not HAS_MPL:
        return False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.arange(LIKERT_MIN - 0.5, LIKERT_MAX + 1.5, 1)
    axes[0].hist(scores, bins=bins, edgecolor="white", color=TAB10_BLUE, alpha=0.85)
    axes[0].set_xticks(range(LIKERT_MIN, LIKERT_MAX + 1))
    axes[0].set_xlabel("评分")
    axes[0].set_ylabel("人数")
    axes[0].set_title("N 列评分分布（直方图）")

    freq = scores.value_counts().sort_index()
    x = list(range(LIKERT_MIN, LIKERT_MAX + 1))
    y = [freq.get(i, 0) for i in x]
    axes[1].bar(x, y, color=TAB10_BLUE, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("评分")
    axes[1].set_ylabel("人数")
    for xi, yi in zip(x, y):
        axes[1].text(xi, yi + 0.5, str(yi), ha="center", va="bottom", fontsize=9)
    axes[1].set_title("N 列各分值人数（柱状图）")

    short_title = col_name if len(col_name) <= 40 else col_name[:37] + "..."
    fig.suptitle(f"N 列分析：{short_title}\n有效样本 n={len(scores)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _apply_reference_axes_style(ax: plt.Axes) -> None:
    """与 Following Style 散点图一致的坐标轴与网格样式。"""
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", color="#cccccc", alpha=0.85)
    ax.axhline(3, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def classify_super_group(group_name: str) -> str | None:
    """将实验轮次归入 l3 / l4 follow / l4 overtake 三组之一。"""
    name = str(group_name).strip().lower()
    if name.startswith("l3"):
        return "l3"
    if "follow" in name:
        return "l4 follow"
    if "overtake" in name:
        return "l4 overtake"
    return None


def extract_driving_style(group_name: str) -> str | None:
    """从组名提取驾驶风格（aggressive / consecutive / neutral / self）。"""
    name = str(group_name).strip().lower()
    for style in DRIVING_STYLES:
        if style in name:
            return style
    return None


def _compute_round_plot_ylim(df: pd.DataFrame) -> float:
    """根据均值+标准差+标注，计算纵轴上限，避免误差线/文字超出图框。"""
    tops: list[float] = []
    for sg_key, _ in SUPER_GROUPS:
        sub = df[df["super_group"] == sg_key]
        for style in DRIVING_STYLES:
            row = sub[sub["style"] == style]
            if row.empty:
                continue
            r0 = row.iloc[0]
            mean_col = "mean" if "mean" in r0.index else "均值"
            std_col = "std" if "std" in r0.index else "标准差"
            m = float(r0[mean_col])
            s = float(r0[std_col]) if pd.notna(r0[std_col]) else 0.0
            tops.append(m + s + Y_LABEL_PAD)
    y_top = max(tops) if tops else float(LIKERT_MAX) + Y_LABEL_PAD
    # 略高于最大误差线，并取 0.25 刻度对齐
    return float(np.ceil(y_top * 4) / 4)


def load_round_csv(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or ROUND_CSV
    if not path.exists():
        raise FileNotFoundError(f"未找到分组统计文件: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    group_col = "group" if "group" in df.columns else "组别"
    if group_col not in df.columns:
        df = df.rename(columns={df.columns[0]: "组别"})
        group_col = "组别"
    if group_col != "group":
        df = df.rename(columns={group_col: "group"})
    df["super_group"] = df["group"].map(classify_super_group)
    df["style"] = df["group"].map(extract_driving_style)
    return df


def build_round_score_detail(df: pd.DataFrame, scores: pd.Series, group_col_keyword: str) -> pd.DataFrame | None:
    """生成每个试次的原始评分明细，用于散点图展示。"""
    candidates = [c for c in df.columns if group_col_keyword in str(c)]
    if not candidates:
        return None

    group_col = candidates[0]
    detail = pd.DataFrame(
        {
            "group": df.loc[scores.index, group_col].values,
            "score": scores.astype(float).values,
        }
    ).dropna(subset=["group", "score"])
    if detail.empty:
        return None

    detail["super_group"] = detail["group"].map(classify_super_group)
    detail["style"] = detail["group"].map(extract_driving_style)
    detail = detail.dropna(subset=["super_group", "style"])
    return detail


def plot_round_score_scatter(detail: pd.DataFrame, out_path: Path | None = None) -> bool:
    """按 L3 / L4 Following / L4 Overtaking 绘制四种风格的原始评分散点图。"""
    if not HAS_MPL:
        return False

    out = out_path or ROUND_SCATTER_FIG
    _setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Naturalness Score Distribution", fontsize=13, fontweight="bold", y=1.02)

    rng = np.random.default_rng(2026)
    for ax, (sg_key, sg_title) in zip(axes, SUPER_GROUPS):
        sub = detail[detail["super_group"] == sg_key].copy()
        x_positions = np.arange(len(DRIVING_STYLES))

        for x_pos, style in zip(x_positions, DRIVING_STYLES):
            values = sub.loc[sub["style"] == style, "score"].astype(float).to_numpy()
            if len(values) == 0:
                continue

            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), x_pos) + jitter,
                values,
                s=34,
                color=STYLE_COLORS[style],
                edgecolors="#333333",
                linewidths=0.45,
                alpha=0.85,
                zorder=3,
                clip_on=True,
            )

            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            ax.scatter(
                [x_pos],
                [mean_value],
                marker="D",
                s=48,
                facecolors="white",
                edgecolors="#333333",
                linewidths=1.0,
                zorder=4,
                clip_on=True,
            )
            ax.text(
                x_pos,
                min(mean_value + 0.18, LIKERT_MAX + 0.25),
                f"M={mean_value:.2f}\nMed={median_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
                clip_on=True,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([STYLE_LABELS[s] for s in DRIVING_STYLES], rotation=15, ha="right")
        ax.set_ylim(LIKERT_MIN - 0.35, LIKERT_MAX + 0.45)
        ax.set_yticks(range(LIKERT_MIN, LIKERT_MAX + 1))
        ax.set_title(sg_title, fontsize=11, fontweight="bold", color="black")
        if ax is axes[0]:
            ax.set_ylabel("naturalness score")
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.axhline(TEST_MU, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_round_super_groups(csv_path: Path | None = None, out_path: Path | None = None) -> bool:
    """
    可视化「按实验轮次」原始评分分布：l3 / l4 follow / l4 overtake 三个子图。
    每个子图使用箱线图 + 原始散点展示四种驾驶风格。
    """
    if not HAS_MPL:
        return False

    from matplotlib.patches import Patch

    out = out_path or ROUND_FIG
    _setup_chinese_font()
    if ROUND_SCORE_CSV.exists():
        score_detail = pd.read_csv(ROUND_SCORE_CSV, encoding="utf-8-sig")
    else:
        score_detail = None
    if score_detail is None or score_detail.empty:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Naturalness Analysis", fontsize=13, fontweight="bold", y=1.03)

    rng = np.random.default_rng(2026)
    for ax, (sg_key, sg_title) in zip(axes, SUPER_GROUPS):
        sub = score_detail[score_detail["super_group"] == sg_key].copy()
        data, labels, colors, plotted_styles = [], [], [], []
        for style in DRIVING_STYLES:
            values = sub.loc[sub["style"] == style, "score"].astype(float).to_numpy()
            if len(values) == 0:
                continue
            data.append(values)
            labels.append(STYLE_LABELS[style])
            colors.append(STYLE_COLORS[style])
            plotted_styles.append(style)

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

        for x_pos, values, style in zip(x_positions, data, plotted_styles):
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
                1.45,
                f"M={mean_value:.2f}\nMed={median_value:.1f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="#333333",
                clip_on=True,
            )

        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(LIKERT_MIN - 0.35, LIKERT_MAX + 0.45)
        ax.tick_params(axis="x", pad=2)
        ax.set_yticks(range(LIKERT_MIN, LIKERT_MAX + 1))
        ax.set_title(sg_title, fontsize=11, fontweight="bold", color="black")
        if ax is axes[0]:
            ax.set_ylabel("average score")
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.axhline(TEST_MU, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=STYLE_COLORS[s], edgecolor="#333333", label=STYLE_LABELS[s])
        for s in DRIVING_STYLES
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
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _format_extended_block(row: dict, indent: str = "  ") -> list[str]:
    lines = [
        f"{indent}均值: {row['均值']}",
        f"{indent}标准差: {row['标准差']}",
        f"{indent}95% CI: [{row['95%CI下限']}, {row['95%CI上限']}]",
        f"{indent}高于 3 分比例: {row['高于3分比例(%)']}%",
        f"{indent}≥4 分比例: {row['大于等于4分比例(%)']}%",
        f"{indent}单样本 t 检验 (H0: μ={row['检验参照值']}): "
        f"t={row['t检验统计量']}, p={row['t检验p值']}",
        f"{indent}Wilcoxon 符号秩检验 (相对 {row['检验参照值']}): "
        f"W={row['Wilcoxon统计量']}, p={row['Wilcoxon p值']}",
    ]
    return lines


def write_report(
    col_name: str,
    raw: pd.Series,
    valid: pd.Series,
    invalid: pd.Series,
    stats: pd.Series,
    freq: pd.DataFrame,
    overall_ext: dict,
    by_round: pd.DataFrame | None,
    out_txt: Path,
) -> None:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("自动驾驶问卷 — N 列数据分析报告")
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
    lines.append(f"  25% 分位: {stats['25%']:.1f}")
    lines.append(f"  75% 分位: {stats['75%']:.1f}")
    lines.append("")
    lines.append("【频数分布】")
    lines.append(freq.to_string(index=False))
    lines.append("")
    lines.append("【说明】")
    lines.append(f"  单样本 t / Wilcoxon 检验以量表中立点 μ={TEST_MU} 为参照，")
    lines.append("  检验评分是否系统性偏离中立。")
    if by_round is not None:
        lines.append("")
        lines.append("【按实验轮次分组 — 完整指标表】")
        lines.append(by_round.to_string(index=False))
    lines.append("")
    lines.append("=" * 60)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, col_name, df = load_column_n()
    valid, invalid = clean_scores(raw)

    if valid.empty:
        print("错误：没有有效的 1–5 分整数数据。", file=sys.stderr)
        return 1

    stats = descriptive_stats(valid)
    freq = frequency_table(valid)
    overall_ext = extended_stats_row(valid, group_name="总体")
    by_round = group_stats(df, valid, "实验轮次")
    round_score_detail = build_round_score_detail(df, valid, "实验轮次")

    report_path = OUTPUT_DIR / "N列分析报告.txt"
    freq_path = OUTPUT_DIR / "N列频数表.csv"
    fig_path = OUTPUT_DIR / "N列分布图.png"

    ext_rows = [overall_ext]
    if by_round is not None:
        ext_rows.extend(by_round.to_dict(orient="records"))
    extended_stats_table(ext_rows).to_csv(EXTENDED_STATS_CSV, index=False, encoding="utf-8-sig")
    if by_round is not None:
        by_round.to_csv(ROUND_CSV, index=False, encoding="utf-8-sig")
    if round_score_detail is not None:
        round_score_detail.to_csv(ROUND_SCORE_CSV, index=False, encoding="utf-8-sig")

    write_report(col_name, raw, valid, invalid, stats, freq, overall_ext, by_round, report_path)
    freq.to_csv(freq_path, index=False, encoding="utf-8-sig")
    plotted = plot_distribution(valid, col_name, fig_path)

    print(f"分析完成。有效样本: {len(valid)}")
    print(f"  报告: {report_path}")
    print(f"  扩展统计: {EXTENDED_STATS_CSV}")
    print(f"  频数表: {freq_path}")
    if plotted:
        print(f"  图表: {fig_path}")
    else:
        print("  图表: 未生成（carla 环境中未安装 matplotlib，可执行: conda install -n carla matplotlib）")
    if by_round is not None:
        print(f"  分组统计: {ROUND_CSV}")
        if plot_round_super_groups():
            print(f"  分组对比图: {ROUND_FIG}")
        elif HAS_MPL:
            print(f"  分组对比图: 未生成（请确认 {ROUND_CSV} 存在）")
        if round_score_detail is not None and plot_round_score_scatter(round_score_detail):
            print(f"  原始评分散点图: {ROUND_SCATTER_FIG}")
            print(f"  原始评分明细: {ROUND_SCORE_CSV}")
    elif ROUND_CSV.exists() and plot_round_super_groups():
        print(f"  分组对比图: {ROUND_FIG}")
    print()
    print(freq.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

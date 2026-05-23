# -*- coding: utf-8 -*-
"""
统计每个被试在 L3 follow 场景下，不同驾驶风格车辆的接管次数。

解析规则：
  - space_presses = []              -> 0 次
  - space_presses = [{...}, {...}]   -> 2 次
  - space_presses = [3]             -> 3 次
  - space_presses = 3               -> 3 次
  - 混合列表中，数字按其数值计数，事件对象按 1 次计数

使用 carla 环境运行：
  C:\\Users\\16638\\miniconda3\\envs\\carla\\python.exe analyze_l3_takeover_counts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "实验数据2"
OUTPUT_DIR = BASE_DIR / "analysis_output_l3_takeovers"

STYLES = ["aggressive", "consecutive", "neutral", "self"]
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
STYLE_COLUMNS = {
    "aggressive": "aggressive接管次数",
    "consecutive": "consecutive接管次数",
    "neutral": "neutral接管次数",
    "self": "self接管次数",
}

JSON_PATTERN = "driving_data_l3_events_residual_gru_takeover_20s_yaw_shrink_controls*.json"


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", subject_id)
    if match:
        return int(match.group()), subject_id
    return 10_000, subject_id


def parse_space_presses(value: Any) -> tuple[int, str]:
    """把不同格式的 space_presses 统一解析为接管次数。"""
    if value is None:
        return 0, "missing/null"

    if isinstance(value, bool):
        return int(value), "bool"

    if isinstance(value, (int, float)):
        return int(value), "number"

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0, "empty string"
        try:
            return int(float(text)), "numeric string"
        except ValueError:
            return 1, "non-empty string"

    if isinstance(value, dict):
        # 极少数情况下若直接存成对象，视作一次事件。
        return 1, "dict event"

    if isinstance(value, list):
        total = 0
        item_types: list[str] = []
        for item in value:
            count, item_type = parse_space_presses(item)
            total += count
            item_types.append(item_type)
        if not value:
            return 0, "empty list"
        return total, "list: " + ", ".join(sorted(set(item_types)))

    return 0, f"unsupported: {type(value).__name__}"


def iter_takeover_json_files() -> list[Path]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到数据目录: {DATA_ROOT}")
    return sorted(DATA_ROOT.glob(f"T*/follow/*/{JSON_PATTERN}"))


def parse_file(path: Path) -> dict[str, Any]:
    parts = path.parts
    # ... / 实验数据2 / T1 / follow / aggressive / file.json
    subject_id = path.parents[2].name
    style = path.parent.name

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_value = data.get("space_presses")
    count, parsed_as = parse_space_presses(raw_value)

    return {
        "被试编号": subject_id,
        "风格": style,
        "接管次数": count,
        "space_presses解析方式": parsed_as,
        "space_presses原始类型": type(raw_value).__name__,
        "文件路径": str(path.relative_to(BASE_DIR)),
    }


def build_tables(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = (
        detail_df.pivot_table(
            index="被试编号",
            columns="风格",
            values="接管次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=STYLES, fill_value=0)
        .reset_index()
    )

    rename_map = {style: STYLE_COLUMNS[style] for style in STYLES}
    wide = wide.rename(columns=rename_map)
    wide["总接管次数"] = wide[list(rename_map.values())].sum(axis=1)
    wide = wide.sort_values("被试编号", key=lambda s: s.map(subject_sort_key)).reset_index(drop=True)

    style_summary = (
        detail_df.groupby("风格")["接管次数"]
        .agg(["count", "sum", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "风格": "风格",
                "count": "文件数",
                "sum": "总接管次数",
                "mean": "平均每被试接管次数",
                "std": "标准差",
                "min": "最小值",
                "max": "最大值",
            }
        )
    )
    style_summary = style_summary.sort_values(
        "风格", key=lambda s: s.map({style: i for i, style in enumerate(STYLES)})
    )
    return wide, style_summary


def plot_style_box_with_points(detail_df: pd.DataFrame, out_path: Path) -> bool:
    """按风格绘制箱线图，并叠加每个被试的接管次数散点。"""
    if not HAS_MPL:
        return False

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data = [
        detail_df.loc[detail_df["风格"] == style, "接管次数"].astype(float).to_numpy()
        for style in STYLES
    ]
    labels = [STYLE_LABELS[style] for style in STYLES]
    colors = [STYLE_COLORS[style] for style in STYLES]

    max_count = max((float(values.max()) for values in data if len(values)), default=0.0)
    y_top = max(5.0, np.ceil((max_count + 2.0) / 2.0) * 2.0)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("L3 Takeover Counts by Driving Style", fontsize=14, fontweight="bold", y=1.02)

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.52,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 6,
        },
        medianprops={"color": "#333333", "linewidth": 1.6},
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
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.9)

    rng = np.random.default_rng(2026)
    for x_pos, (values, color) in enumerate(zip(data, colors), start=1):
        jitter_x = rng.normal(x_pos, 0.055, size=len(values))
        ax.scatter(
            jitter_x,
            values,
            s=34,
            color=color,
            edgecolors="#333333",
            linewidths=0.45,
            alpha=0.85,
            zorder=3,
            clip_on=True,
        )
        mean_value = float(np.mean(values)) if len(values) else 0.0
        median_value = float(np.median(values)) if len(values) else 0.0
        label_y = min(max(values.max() if len(values) else 0.0, mean_value) + 0.45, y_top - 0.25)
        ax.text(
            x_pos,
            label_y,
            f"Mean={mean_value:.2f}\nMedian={median_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            clip_on=True,
        )

    ax.set_ylabel("踏板触发次数")
    ax.set_ylim(-0.5, y_top)
    ax.set_yticks(np.arange(0, y_top + 1, 2))
    ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_outputs(detail_df: pd.DataFrame, wide_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wide_path = OUTPUT_DIR / "L3接管次数_按被试宽表.csv"
    detail_path = OUTPUT_DIR / "L3接管次数_明细表.csv"
    summary_path = OUTPUT_DIR / "L3接管次数_按风格汇总.csv"
    xlsx_path = OUTPUT_DIR / "L3接管次数汇总.xlsx"
    boxplot_path = OUTPUT_DIR / "L3接管次数_箱线图_带散点.png"
    report_path = OUTPUT_DIR / "L3接管次数_报告.txt"

    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plotted = plot_style_box_with_points(detail_df, boxplot_path)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="按被试宽表", index=False)
        detail_df.to_excel(writer, sheet_name="明细表", index=False)
        summary_df.to_excel(writer, sheet_name="按风格汇总", index=False)

    lines = [
        "L3 车辆接管次数统计",
        "=" * 60,
        f"数据目录: {DATA_ROOT}",
        f"JSON 文件数: {len(detail_df)}",
        f"被试数: {wide_df['被试编号'].nunique()}",
        "",
        "【按风格汇总】",
        summary_df.to_string(index=False),
        "",
        "【输出文件】",
        f"宽表: {wide_path}",
        f"明细表: {detail_path}",
        f"风格汇总: {summary_path}",
        f"Excel汇总: {xlsx_path}",
    ]
    if plotted:
        lines.append(f"箱线图: {boxplot_path}")
    else:
        lines.append("箱线图: 未生成（当前环境未安装 matplotlib）")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    files = iter_takeover_json_files()
    if not files:
        print(f"未找到 JSON 文件: {DATA_ROOT}", file=sys.stderr)
        return 1

    rows = [parse_file(path) for path in files]
    detail_df = pd.DataFrame(rows)
    detail_df = detail_df[detail_df["风格"].isin(STYLES)].copy()
    detail_df = detail_df.sort_values(
        ["被试编号", "风格"],
        key=lambda col: col.map(subject_sort_key) if col.name == "被试编号" else col,
    ).reset_index(drop=True)

    wide_df, summary_df = build_tables(detail_df)
    write_outputs(detail_df, wide_df, summary_df)

    print(f"解析完成：{len(detail_df)} 个文件，{wide_df['被试编号'].nunique()} 名被试。")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print(wide_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

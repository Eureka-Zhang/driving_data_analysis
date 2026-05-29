# -*- coding: utf-8 -*-
r"""
Analyze whether style-label groups prefer the corresponding vehicle style.

The group labels are read from style.txt:
  - Following Label is used for L3 Following and L4 Following
  - Overtaking Label is used for L4 Overtaking

Outputs:
  - label group x vehicle style mean-rating tables
  - label group x preferred vehicle style matrices
  - alignment summaries
  - heatmaps for following and overtaking separately

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_label_preference_alignment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from analyze_h3_l4_task_preference import (
    BASE_DIR,
    EXCEL_FILE,
    GROUP_COL_INDEX,
    METRICS,
    PREFERENCE_METRIC_LABELS,
    PREFERENCE_METRICS,
    SUBJECT_COL_INDEX,
    STYLE_LABELS,
    STYLE_ORDER,
    build_acceptance_composite,
    extract_style,
    setup_font,
)


STYLE_FILE = BASE_DIR / "style.txt"
OUTPUT_DIR = BASE_DIR / "analysis_output_label_preference_alignment"

DISPLAY_STYLES = [STYLE_LABELS[s] for s in STYLE_ORDER]
LABEL_TO_STYLE = {
    "Aggressive": "Aggressive",
    "Conservative": "Conservative",
    "Neutral": "Neutral",
    "Self": "Self",
}
ALL_TASKS = [
    ("l3", "L3 Following"),
    ("l4 follow", "L4 Following"),
    ("l4 overtake", "L4 Overtaking"),
]
TASK_LABEL_COLUMNS = {
    "l3": "Following Label",
    "l4 follow": "Following Label",
    "l4 overtake": "Overtaking Label",
}


def task_display(task_key: str) -> str:
    return dict(ALL_TASKS).get(task_key, task_key)


def classify_task(group_name: str) -> str | None:
    name = str(group_name).strip().lower()
    if name.startswith("l3"):
        return "l3"
    if "l4" not in name:
        return None
    if "follow" in name:
        return "l4 follow"
    if "overtake" in name:
        return "l4 overtake"
    return None


def load_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Load L3 + L4 following/overtaking trial-level ratings."""
    detail = pd.DataFrame(
        {
            "subject": df.iloc[:, SUBJECT_COL_INDEX],
            "group": df.iloc[:, GROUP_COL_INDEX],
        }
    )
    detail["task"] = detail["group"].map(classify_task)
    detail["style"] = detail["group"].map(extract_style)
    detail = detail.dropna(subset=["subject", "group", "task", "style"]).copy()

    for metric in METRICS:
        if metric.key == "comfort_rank":
            continue
        score = pd.to_numeric(df.iloc[:, metric.column_index], errors="coerce")
        valid = score.between(metric.y_min, metric.y_max)
        if metric.integer_only:
            valid &= (score % 1 == 0)
        detail[metric.key] = score.where(valid)

    detail["comfort_rank"] = detail.groupby(["subject", "task"])["comfort"].rank(method="average")
    detail["acceptance_composite"] = build_acceptance_composite(detail)
    return detail


def load_style_labels() -> pd.DataFrame:
    if not STYLE_FILE.exists():
        raise FileNotFoundError(f"未找到标签文件: {STYLE_FILE}")

    labels = pd.read_csv(STYLE_FILE, sep="\t")
    required = {"Driver", "Following Label", "Overtaking Label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"style.txt 缺少列: {sorted(missing)}")

    labels = labels.rename(columns={"Driver": "subject"}).copy()
    for col in ["Following Label", "Overtaking Label"]:
        labels[col] = labels[col].astype(str).str.strip()
    labels["subject"] = labels["subject"].astype(str).str.strip()
    return labels


def attach_labels(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = detail.merge(labels, on="subject", how="left")
    merged["task_label"] = merged.apply(
        lambda row: row[TASK_LABEL_COLUMNS[row["task"]]] if row["task"] in TASK_LABEL_COLUMNS else np.nan,
        axis=1,
    )
    merged["vehicle_style"] = merged["style"].map(STYLE_LABELS)
    return merged.dropna(subset=["task_label", "vehicle_style"]).copy()


def label_style_mean_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_key, task_name in ALL_TASKS:
        task = detail[detail["task"] == task_key]
        for metric_key in PREFERENCE_METRICS:
            metric_label = PREFERENCE_METRIC_LABELS.get(metric_key, metric_key)
            for label in sorted(task["task_label"].dropna().unique()):
                sub_label = task[task["task_label"] == label]
                means = {}
                ns = {}
                for style in STYLE_ORDER:
                    style_label = STYLE_LABELS[style]
                    values = sub_label.loc[sub_label["style"] == style, metric_key].dropna()
                    means[style_label] = round(float(values.mean()), 4) if not values.empty else np.nan
                    ns[style_label] = int(values.size)
                valid_means = {k: v for k, v in means.items() if pd.notna(v)}
                best_style = max(valid_means, key=valid_means.get) if valid_means else np.nan
                expected_style = LABEL_TO_STYLE.get(label, label)
                rows.append(
                    {
                        "task": task_name,
                        "task_key": task_key,
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "label_group": label,
                        "n_subjects": int(sub_label["subject"].nunique()),
                        "expected_matching_style": expected_style,
                        "best_mean_style": best_style,
                        "mean_best_matches_label": bool(best_style == expected_style) if isinstance(best_style, str) else False,
                        **{f"mean_{style}": means[style] for style in DISPLAY_STYLES},
                        **{f"n_{style}": ns[style] for style in DISPLAY_STYLES},
                    }
                )
    return pd.DataFrame(rows)


def subject_winner_matrix(detail: pd.DataFrame, task_key: str, metric_key: str) -> pd.DataFrame:
    task = detail[detail["task"] == task_key].copy()
    label_col = TASK_LABEL_COLUMNS[task_key]
    wide = task.pivot_table(index="subject", columns="style", values=metric_key, aggfunc="mean")
    labels = task[["subject", label_col]].drop_duplicates("subject").set_index("subject")[label_col]

    all_labels = sorted(labels.dropna().unique())
    matrix = pd.DataFrame(0.0, index=all_labels, columns=DISPLAY_STYLES)
    detail_rows = []
    for subject in sorted(set(wide.index) & set(labels.index)):
        label = labels.loc[subject]
        row = wide.loc[subject]
        max_value = row.max()
        winners = [
            style
            for style in STYLE_ORDER
            if style in row.index and pd.notna(row[style]) and np.isclose(row[style], max_value)
        ]
        if not winners or pd.isna(label):
            continue
        weight = 1.0 / len(winners)
        winner_labels = [STYLE_LABELS[s] for s in winners]
        expected_style = LABEL_TO_STYLE.get(label, label)
        matches_label = expected_style in winner_labels
        prefers_self_vehicle = "self" in winners
        matches_self_style = matches_label or prefers_self_vehicle

        for style in winners:
            matrix.loc[label, STYLE_LABELS[style]] += weight
        detail_rows.append(
            {
                "subject": subject,
                "task": task_display(task_key),
                "task_key": task_key,
                "metric": PREFERENCE_METRIC_LABELS.get(metric_key, metric_key),
                "metric_key": metric_key,
                "label": label,
                "preferred_style": "/".join(winner_labels),
                "matches_label": matches_label,
                "prefers_self_vehicle": prefers_self_vehicle,
                "matches_self_style": matches_self_style,
                "tie_count": len(winners),
            }
        )
    return matrix, pd.DataFrame(detail_rows)


def build_alignment_outputs(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    summary_rows = []
    winner_detail_tables = []
    matrices: dict[tuple[str, str], pd.DataFrame] = {}

    for task_key, task_name in ALL_TASKS:
        for metric_key in PREFERENCE_METRICS:
            metric_label = PREFERENCE_METRIC_LABELS.get(metric_key, metric_key)
            matrix, winner_detail = subject_winner_matrix(detail, task_key, metric_key)
            matrices[(task_key, metric_key)] = matrix
            if not winner_detail.empty:
                winner_detail_tables.append(winner_detail)

            total = float(matrix.to_numpy().sum())
            matched = 0.0
            for label in matrix.index:
                expected = LABEL_TO_STYLE.get(label, label)
                if expected in matrix.columns:
                    matched += float(matrix.loc[label, expected])

            self_vehicle_mass = float(matrix["Self"].sum()) if "Self" in matrix.columns else 0.0
            self_style_aligned = (
                float(winner_detail["matches_self_style"].sum()) if not winner_detail.empty else 0.0
            )

            summary_rows.append(
                {
                    "task": task_name,
                    "task_key": task_key,
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "total_subject_mass": round(total, 4),
                    "matched_subject_mass": round(matched, 4),
                    "matched_percent": round(100 * matched / total, 2) if total else np.nan,
                    "self_vehicle_preferred_mass": round(self_vehicle_mass, 4),
                    "self_vehicle_preferred_percent": round(100 * self_vehicle_mass / total, 2) if total else np.nan,
                    "self_style_aligned_mass": round(self_style_aligned, 4),
                    "self_style_aligned_percent": round(100 * self_style_aligned / total, 2) if total else np.nan,
                }
            )

    winner_detail_all = pd.concat(winner_detail_tables, ignore_index=True) if winner_detail_tables else pd.DataFrame()
    return pd.DataFrame(summary_rows), winner_detail_all, matrices


def plot_matrix(matrix: pd.DataFrame, out_path: Path, title: str) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    values = matrix.reindex(columns=DISPLAY_STYLES).fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_xlabel("Actually preferred vehicle style")
    ax.set_ylabel("Style label group")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values.iloc[i, j])
            label = f"{value:.1f}" if abs(value - round(value)) > 1e-8 else f"{int(round(value))}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, color="#222222")

    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("participants (fractional ties)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_mean_table(mean_table: pd.DataFrame, out_path: Path, task_key: str, metric_key: str) -> bool:
    if not HAS_MPL:
        return False
    setup_font()

    sub = mean_table[(mean_table["task_key"] == task_key) & (mean_table["metric_key"] == metric_key)].copy()
    if sub.empty:
        return False
    labels = sub["label_group"].tolist()
    values = sub[[f"mean_{style}" for style in DISPLAY_STYLES]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor("white")
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(DISPLAY_STYLES)))
    ax.set_xticklabels(DISPLAY_STYLES, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Vehicle style")
    ax.set_ylabel("Style label group")
    ax.set_title(
        f"{task_display(task_key)}: mean {PREFERENCE_METRIC_LABELS.get(metric_key, metric_key)} by label",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("mean rating", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(mean_table: pd.DataFrame, alignment: pd.DataFrame) -> None:
    lines = []
    lines.append("Style-label group preference alignment")
    lines.append("=" * 72)
    lines.append("Labels come from style.txt.")
    lines.append("Following Label is used for L3 Following and L4 Following.")
    lines.append("Overtaking Label is used for L4 Overtaking.")
    lines.append("")
    lines.append("Alignment definitions:")
    lines.append("  matched_percent: preferred vehicle style equals style.txt label (Aggressive/Conservative/Neutral).")
    lines.append("  self_vehicle_preferred_percent: preferred vehicle style is Self.")
    lines.append("  self_style_aligned_percent: label match OR preferred Self vehicle (includes self-style preference).")
    lines.append("")
    lines.append("【Subject-level alignment】")
    lines.append(alignment.to_string(index=False))
    lines.append("")
    lines.append("【Group mean best-style check】")
    cols = [
        "task",
        "metric",
        "label_group",
        "n_subjects",
        "expected_matching_style",
        "best_mean_style",
        "mean_best_matches_label",
    ]
    lines.append(mean_table[cols].to_string(index=False))
    (OUTPUT_DIR / "label_preference_alignment_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(EXCEL_FILE)
    detail = load_detail(df)
    labels = load_style_labels()
    detail_labeled = attach_labels(detail, labels)

    mean_table = label_style_mean_table(detail_labeled)
    alignment, winner_detail, matrices = build_alignment_outputs(detail_labeled)

    detail_labeled.to_csv(OUTPUT_DIR / "label_preference_detail.csv", index=False, encoding="utf-8-sig")
    mean_table.to_csv(OUTPUT_DIR / "label_group_vehicle_style_mean_ratings.csv", index=False, encoding="utf-8-sig")
    alignment.to_csv(OUTPUT_DIR / "label_preference_alignment_summary.csv", index=False, encoding="utf-8-sig")
    winner_detail.to_csv(OUTPUT_DIR / "label_preference_subject_winners.csv", index=False, encoding="utf-8-sig")

    for (task_key, metric_key), matrix in matrices.items():
        task_slug = task_key.replace(" ", "_")
        matrix.to_csv(
            OUTPUT_DIR / f"label_to_preferred_style_{task_slug}_{metric_key}.csv",
            encoding="utf-8-sig",
        )
        plot_matrix(
            matrix,
            OUTPUT_DIR / f"label_to_preferred_style_{task_slug}_{metric_key}.png",
            f"{task_display(task_key)}: label -> preferred style ({PREFERENCE_METRIC_LABELS.get(metric_key, metric_key)})",
        )

    for task_key, _ in ALL_TASKS:
        for metric_key in ["expectation", "acceptance_composite"]:
            plot_mean_table(
                mean_table,
                OUTPUT_DIR / f"label_group_mean_ratings_{task_key.replace(' ', '_')}_{metric_key}.png",
                task_key,
                metric_key,
            )

    write_summary(mean_table, alignment)
    print(f"detail rows: {len(detail_labeled)}")
    print(f"outputs: {OUTPUT_DIR}")
    print()
    print(alignment.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

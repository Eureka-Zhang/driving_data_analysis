# -*- coding: utf-8 -*-
r"""
Analyze whether L3 following takeover counts differ by subject style label
and vehicle trajectory style.

Inputs:
  - style.txt
  - ./实验数据2/T*/follow/<style>/driving_data_l3_events_*.json

Outputs:
  - analysis_output_l3_takeover_label_alignment/*.csv
  - analysis_output_l3_takeover_label_alignment/*.png
  - analysis_output_l3_takeover_label_alignment/result_summary.md

Run:
  C:\Users\16638\miniconda3\envs\carla\python.exe analyze_l3_takeover_label_alignment.py
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.stats import binomtest, friedmanchisquare, kruskal, wilcoxon

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "实验数据2"
STYLE_FILE = BASE_DIR / "style.txt"
OUTPUT_DIR = BASE_DIR / "analysis_output_l3_takeover_label_alignment"

JSON_PATTERN = "driving_data_l3_events_residual_gru_takeover_20s_yaw_shrink_controls*.json"

STYLE_ORDER = ["aggressive", "neutral", "consecutive", "self"]
PRESET_STYLES = ["aggressive", "neutral", "consecutive"]
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
}
LABEL_ORDER = ["Aggressive", "Neutral", "Conservative"]


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", str(subject_id))
    if match:
        return int(match.group()), str(subject_id)
    return 10_000, str(subject_id)


def parse_space_presses(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 1
    if isinstance(value, dict):
        return 1
    if isinstance(value, list):
        return sum(parse_space_presses(item) for item in value)
    return 0


def iter_takeover_json_files() -> list[Path]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_ROOT}")
    return sorted(DATA_ROOT.glob(f"T*/follow/*/{JSON_PATTERN}"))


def parse_takeover_file(path: Path) -> dict[str, Any]:
    subject = path.parents[2].name
    style = path.parent.name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "subject": subject,
        "vehicle_style_key": style,
        "vehicle_style": STYLE_LABELS.get(style, style),
        "takeover_count": parse_space_presses(data.get("space_presses")),
        "file": str(path.relative_to(BASE_DIR)),
    }


def load_takeover_detail() -> pd.DataFrame:
    rows = [parse_takeover_file(path) for path in iter_takeover_json_files()]
    if not rows:
        raise FileNotFoundError(f"No L3 takeover JSON files matched under {DATA_ROOT}")
    detail = pd.DataFrame(rows)
    detail = detail[detail["vehicle_style_key"].isin(STYLE_ORDER)].copy()
    detail["subject"] = detail["subject"].astype(str).str.strip()
    return detail.sort_values(
        ["subject", "vehicle_style_key"],
        key=lambda col: col.map(subject_sort_key) if col.name == "subject" else col,
    ).reset_index(drop=True)


def load_style_labels() -> pd.DataFrame:
    if not STYLE_FILE.exists():
        raise FileNotFoundError(f"style.txt not found: {STYLE_FILE}")
    labels = pd.read_csv(STYLE_FILE, sep="\t")
    required = {"Driver", "Following Label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"style.txt is missing columns: {sorted(missing)}")
    labels = labels.rename(columns={"Driver": "subject"}).copy()
    labels["subject"] = labels["subject"].astype(str).str.strip()
    labels["following_label"] = labels["Following Label"].astype(str).str.strip()
    labels["own_style_key"] = labels["following_label"].map(LABEL_TO_STYLE_KEY)
    return labels[["subject", "following_label", "own_style_key"]].dropna()


def attach_labels(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = detail.merge(labels, on="subject", how="inner")
    merged["own_style"] = merged["own_style_key"].map(STYLE_LABELS)
    merged["is_self_vehicle"] = merged["vehicle_style_key"].eq("self")
    merged["is_own_label_preset"] = merged["vehicle_style_key"].eq(merged["own_style_key"])
    merged["relation_to_subject"] = np.select(
        [merged["is_self_vehicle"], merged["is_own_label_preset"]],
        ["Self vehicle", "Own-label preset"],
        default="Mismatch preset",
    )
    return merged


def make_wide(detail: pd.DataFrame) -> pd.DataFrame:
    wide = detail.pivot_table(
        index=["subject", "following_label", "own_style_key"],
        columns="vehicle_style_key",
        values="takeover_count",
        aggfunc="sum",
    ).reindex(columns=STYLE_ORDER)
    wide = wide.reset_index()
    wide = wide.sort_values("subject", key=lambda s: s.map(subject_sort_key)).reset_index(drop=True)
    return wide


def mismatch_styles(own_style_key: str) -> list[str]:
    return [style for style in PRESET_STYLES if style != own_style_key]


def extreme_mismatch_styles(own_style_key: str) -> list[str]:
    if own_style_key == "aggressive":
        return ["consecutive"]
    if own_style_key == "consecutive":
        return ["aggressive"]
    return ["aggressive", "consecutive"]


def build_subject_contrasts(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in wide.iterrows():
        own_key = row["own_style_key"]
        mismatches = mismatch_styles(own_key)
        extremes = extreme_mismatch_styles(own_key)
        self_count = float(row["self"])
        own_count = float(row[own_key])
        mismatch_mean = float(row[mismatches].mean())
        extreme_mean = float(row[extremes].mean())
        aligned_mean = float(np.mean([self_count, own_count]))
        aligned_min = float(np.min([self_count, own_count]))
        rows.append(
            {
                "subject": row["subject"],
                "following_label": row["following_label"],
                "own_style_key": own_key,
                "own_style": STYLE_LABELS[own_key],
                "self_count": self_count,
                "own_label_preset_count": own_count,
                "aligned_mean_count": aligned_mean,
                "aligned_min_count": aligned_min,
                "mismatch_mean_count": mismatch_mean,
                "extreme_mismatch_count": extreme_mean,
                "mismatch_styles": "/".join(STYLE_LABELS[s] for s in mismatches),
                "extreme_mismatch_styles": "/".join(STYLE_LABELS[s] for s in extremes),
                "self_minus_mismatch": self_count - mismatch_mean,
                "own_minus_mismatch": own_count - mismatch_mean,
                "aligned_mean_minus_mismatch": aligned_mean - mismatch_mean,
                "aligned_min_minus_mismatch": aligned_min - mismatch_mean,
            }
        )
    return pd.DataFrame(rows)


def descriptives(detail: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_style = (
        detail.groupby(["following_label", "vehicle_style_key", "vehicle_style"])["takeover_count"]
        .agg(["count", "mean", "std", "median", "min", "max", "sum"])
        .reset_index()
        .rename(columns={"count": "n_observations"})
    )
    group_style["n_subjects"] = group_style["n_observations"]
    group_style = group_style.sort_values(
        ["following_label", "vehicle_style_key"],
        key=lambda col: col.map({s: i for i, s in enumerate(STYLE_ORDER)})
        if col.name == "vehicle_style_key"
        else col.map({label: i for i, label in enumerate(LABEL_ORDER)}),
    )

    contrast_desc = (
        contrasts.groupby("following_label")[
            [
                "self_count",
                "own_label_preset_count",
                "aligned_mean_count",
                "aligned_min_count",
                "mismatch_mean_count",
                "extreme_mismatch_count",
                "self_minus_mismatch",
                "own_minus_mismatch",
                "aligned_mean_minus_mismatch",
                "aligned_min_minus_mismatch",
            ]
        ]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    contrast_desc.columns = [
        "_".join(str(part) for part in col if part).rstrip("_") for col in contrast_desc.columns.to_flat_index()
    ]
    return group_style, contrast_desc


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


def rank_biserial_lower_advantage(diff: pd.Series) -> float:
    nonzero = diff[diff != 0].dropna()
    if nonzero.empty:
        return np.nan
    ranks = nonzero.abs().rank(method="average")
    w_neg = float(ranks[nonzero < 0].sum())
    w_pos = float(ranks[nonzero > 0].sum())
    denom = w_neg + w_pos
    if denom == 0:
        return np.nan
    return (w_neg - w_pos) / denom


def paired_lower_test(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    pair = pd.concat([left, right], axis=1, keys=["left", "right"]).dropna()
    diff = pair["left"] - pair["right"]
    n = int(len(diff))
    n_less = int((diff < 0).sum())
    n_equal = int((diff == 0).sum())
    n_greater = int((diff > 0).sum())
    nonzero_n = n_less + n_greater

    if n == 0:
        return {
            "n": 0,
            "left_mean": np.nan,
            "right_mean": np.nan,
            "mean_diff_left_minus_right": np.nan,
            "median_diff_left_minus_right": np.nan,
            "n_left_lower": 0,
            "n_equal": 0,
            "n_left_higher": 0,
            "wilcoxon_w": np.nan,
            "p_wilcoxon_less": np.nan,
            "p_wilcoxon_two_sided": np.nan,
            "p_sign_less": np.nan,
            "rank_biserial_lower_advantage": np.nan,
        }

    if nonzero_n == 0 or not HAS_SCIPY:
        wilcox_w = 0.0 if nonzero_n == 0 else np.nan
        p_less = 1.0 if nonzero_n == 0 else np.nan
        p_two = 1.0 if nonzero_n == 0 else np.nan
        p_sign = 1.0 if nonzero_n == 0 else np.nan
    else:
        less_result = wilcoxon(pair["left"], pair["right"], alternative="less", zero_method="wilcox")
        two_result = wilcoxon(pair["left"], pair["right"], alternative="two-sided", zero_method="wilcox")
        wilcox_w = float(less_result.statistic)
        p_less = float(less_result.pvalue)
        p_two = float(two_result.pvalue)
        p_sign = float(binomtest(n_less, nonzero_n, 0.5, alternative="greater").pvalue)

    return {
        "n": n,
        "left_mean": round(float(pair["left"].mean()), 4),
        "right_mean": round(float(pair["right"].mean()), 4),
        "mean_diff_left_minus_right": round(float(diff.mean()), 4),
        "median_diff_left_minus_right": round(float(diff.median()), 4),
        "n_left_lower": n_less,
        "n_equal": n_equal,
        "n_left_higher": n_greater,
        "wilcoxon_w": round(wilcox_w, 4) if pd.notna(wilcox_w) else np.nan,
        "p_wilcoxon_less": round(p_less, 6) if pd.notna(p_less) else np.nan,
        "p_wilcoxon_two_sided": round(p_two, 6) if pd.notna(p_two) else np.nan,
        "p_sign_less": round(p_sign, 6) if pd.notna(p_sign) else np.nan,
        "rank_biserial_lower_advantage": round(rank_biserial_lower_advantage(diff), 4),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return q
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * m / rank)
        adjusted[i] = value
        prev = value
    q.loc[order] = np.clip(adjusted, 0, 1)
    return q


def contrast_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("self_count", "mismatch_mean_count", "Self - mismatch preset mean"),
        ("own_label_preset_count", "mismatch_mean_count", "Own-label preset - mismatch preset mean"),
        ("aligned_mean_count", "mismatch_mean_count", "Mean(Self, own-label preset) - mismatch preset mean"),
        ("aligned_min_count", "mismatch_mean_count", "Min(Self, own-label preset) - mismatch preset mean"),
        ("self_count", "extreme_mismatch_count", "Self - extreme mismatch"),
        ("own_label_preset_count", "extreme_mismatch_count", "Own-label preset - extreme mismatch"),
        ("self_count", "own_label_preset_count", "Self - own-label preset"),
    ]
    scopes: list[tuple[str, pd.DataFrame]] = [("All subjects", contrasts)]
    for label in LABEL_ORDER:
        part = contrasts[contrasts["following_label"] == label]
        if not part.empty:
            scopes.append((label, part))

    rows = []
    for scope, data in scopes:
        for left_col, right_col, comparison in comparisons:
            row = {
                "scope": scope,
                "comparison": comparison,
                "left_column": left_col,
                "right_column": right_col,
                **paired_lower_test(data[left_col], data[right_col]),
            }
            rows.append(row)

    tests = pd.DataFrame(rows)
    tests["q_wilcoxon_less_by_scope"] = tests.groupby("scope")["p_wilcoxon_less"].transform(benjamini_hochberg)
    return tests


def style_pairwise_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("All subjects", wide)]
    for label in LABEL_ORDER:
        part = wide[wide["following_label"] == label]
        if not part.empty:
            scopes.append((label, part))

    for scope, data in scopes:
        for left, right in combinations(STYLE_ORDER, 2):
            rows.append(
                {
                    "scope": scope,
                    "comparison": f"{STYLE_LABELS[left]} - {STYLE_LABELS[right]}",
                    "left_style": left,
                    "right_style": right,
                    **paired_lower_test(data[left], data[right]),
                }
            )

    tests = pd.DataFrame(rows)
    tests["q_wilcoxon_less_by_scope"] = tests.groupby("scope")["p_wilcoxon_less"].transform(benjamini_hochberg)
    return tests


def friedman_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("All subjects", wide)]
    for label in LABEL_ORDER:
        part = wide[wide["following_label"] == label]
        if not part.empty:
            scopes.append((label, part))

    for scope, data in scopes:
        if not HAS_SCIPY or len(data) < 3:
            stat, p_value = np.nan, np.nan
        else:
            values = [data[style].astype(float).to_numpy() for style in STYLE_ORDER]
            stat, p_value = friedmanchisquare(*values)
        rows.append(
            {
                "scope": scope,
                "n_subjects": int(len(data)),
                "test": "Friedman vehicle-style effect",
                "statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                "p_value": round(float(p_value), 6) if pd.notna(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subject_dummy_matrix(subjects: np.ndarray) -> np.ndarray:
    levels = list(dict.fromkeys(subjects))
    return np.column_stack([(subjects == level).astype(float) for level in levels])


def style_dummy_matrix(styles: np.ndarray) -> np.ndarray:
    return np.column_stack([(styles == style).astype(float) for style in STYLE_ORDER[1:]])


def interaction_matrix(labels: np.ndarray, styles: np.ndarray) -> np.ndarray:
    columns = []
    for label in LABEL_ORDER[1:]:
        for style in STYLE_ORDER[1:]:
            columns.append(((labels == label) & (styles == style)).astype(float))
    return np.column_stack(columns) if columns else np.empty((len(styles), 0))


def rss_and_rank(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(np.sum(resid**2)), int(rank)


def f_from_models(y: np.ndarray, x_reduced: np.ndarray, x_full: np.ndarray) -> tuple[float, int, int]:
    rss_reduced, rank_reduced = rss_and_rank(y, x_reduced)
    rss_full, rank_full = rss_and_rank(y, x_full)
    df_num = rank_full - rank_reduced
    df_den = len(y) - rank_full
    if df_num <= 0 or df_den <= 0 or rss_full <= 0:
        return np.nan, df_num, df_den
    f_value = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
    return float(max(f_value, 0.0)), df_num, df_den


def long_arrays_from_wide(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_values = []
    subjects = []
    labels = []
    styles = []
    for _, row in wide.iterrows():
        for style in STYLE_ORDER:
            y_values.append(float(row[style]))
            subjects.append(row["subject"])
            labels.append(row["following_label"])
            styles.append(style)
    return (
        np.asarray(y_values, dtype=float),
        np.asarray(subjects, dtype=object),
        np.asarray(labels, dtype=object),
        np.asarray(styles, dtype=object),
    )


def permutation_style_main(wide: pd.DataFrame, n_perm: int = 20_000, seed: int = 20260530) -> dict[str, Any]:
    y, subjects, _, styles = long_arrays_from_wide(wide)
    x_subject = subject_dummy_matrix(subjects)
    x_style = np.column_stack([x_subject, style_dummy_matrix(styles)])
    observed, df_num, df_den = f_from_models(y, x_subject, x_style)

    rng = np.random.default_rng(seed)
    count_ge = 0
    matrix = wide[STYLE_ORDER].astype(float).to_numpy()
    for _ in range(n_perm):
        shuffled = np.vstack([rng.permutation(row) for row in matrix])
        y_perm = shuffled.reshape(-1)
        stat, _, _ = f_from_models(y_perm, x_subject, x_style)
        if pd.notna(stat) and stat >= observed - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {
        "test": "Permutation repeated-measures style main effect",
        "statistic": round(observed, 4),
        "df_num": df_num,
        "df_den": df_den,
        "n_permutations": n_perm,
        "p_value": round(p_value, 6),
    }


def permutation_label_style_interaction(
    wide: pd.DataFrame, n_perm: int = 20_000, seed: int = 20260531
) -> dict[str, Any]:
    y, subjects, labels, styles = long_arrays_from_wide(wide)
    x_subject_style = np.column_stack([subject_dummy_matrix(subjects), style_dummy_matrix(styles)])
    x_full = np.column_stack([x_subject_style, interaction_matrix(labels, styles)])
    observed, df_num, df_den = f_from_models(y, x_subject_style, x_full)

    rng = np.random.default_rng(seed)
    subject_table = wide[["subject", "following_label"]].drop_duplicates().reset_index(drop=True)
    original_labels = subject_table["following_label"].to_numpy()
    subject_to_position = {subject: i for i, subject in enumerate(subject_table["subject"])}
    subject_positions_long = np.asarray([subject_to_position[subject] for subject in subjects])

    count_ge = 0
    for _ in range(n_perm):
        permuted_subject_labels = rng.permutation(original_labels)
        permuted_labels = permuted_subject_labels[subject_positions_long]
        x_perm = np.column_stack([x_subject_style, interaction_matrix(permuted_labels, styles)])
        stat, _, _ = f_from_models(y, x_subject_style, x_perm)
        if pd.notna(stat) and stat >= observed - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {
        "test": "Permutation following-label x vehicle-style interaction",
        "statistic": round(observed, 4),
        "df_num": df_num,
        "df_den": df_den,
        "n_permutations": n_perm,
        "p_value": round(p_value, 6),
    }


def mixed_design_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = [permutation_style_main(wide), permutation_label_style_interaction(wide)]
    if HAS_SCIPY:
        grouped = [
            wide.loc[wide["following_label"] == label, STYLE_ORDER].astype(float).to_numpy().mean(axis=1)
            for label in LABEL_ORDER
            if not wide.loc[wide["following_label"] == label].empty
        ]
        if len(grouped) >= 2:
            stat, p_value = kruskal(*grouped)
            rows.append(
                {
                    "test": "Kruskal following-label main effect on subject mean count",
                    "statistic": round(float(stat), 4),
                    "df_num": len(grouped) - 1,
                    "df_den": np.nan,
                    "n_permutations": 0,
                    "p_value": round(float(p_value), 6),
                }
            )
    return pd.DataFrame(rows)


def plot_group_style_means(group_style: pd.DataFrame, detail: pd.DataFrame, out_path: Path) -> bool:
    """Boxplots by following-label group, styled after analyze_l3_takeover_counts.py."""
    if not HAS_MPL:
        return False

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    all_values = detail["takeover_count"].astype(float)
    max_count = float(all_values.max()) if not all_values.empty else 0.0
    y_top = max(5.0, np.ceil((max_count + 2.0) / 2.0) * 2.0)

    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(13.5, 5.8), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("L3 Takeover Counts by Following Label and Vehicle Style", fontsize=14, fontweight="bold")

    rng = np.random.default_rng(2026)
    colors = [STYLE_COLORS[style] for style in STYLE_ORDER]
    style_labels = [STYLE_LABELS[style] for style in STYLE_ORDER]
    group_ns = detail.groupby("following_label")["subject"].nunique().reindex(LABEL_ORDER)

    for ax, label in zip(axes, LABEL_ORDER):
        panel_data = [
            detail.loc[
                (detail["following_label"] == label) & (detail["vehicle_style_key"] == style),
                "takeover_count",
            ]
            .astype(float)
            .to_numpy()
            for style in STYLE_ORDER
        ]

        box = ax.boxplot(
            panel_data,
            tick_labels=style_labels,
            patch_artist=True,
            widths=0.52,
            showmeans=True,
            showfliers=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 5.5,
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

        for x_pos, (values, color) in enumerate(zip(panel_data, colors), start=1):
            jitter_x = rng.normal(x_pos, 0.055, size=len(values))
            ax.scatter(
                jitter_x,
                values,
                s=32,
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
                fontsize=8,
                color="#333333",
                clip_on=True,
            )

        ax.set_title(f"{label} label (n={int(group_ns.loc[label])})", fontsize=11, fontweight="bold")
        ax.set_xticklabels(style_labels, rotation=18, ha="right")
        ax.set_ylim(-0.5, y_top)
        ax.set_yticks(np.arange(0, y_top + 1, 2))
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("L3 takeover count")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_contrast_lines(contrasts: pd.DataFrame, out_path: Path) -> bool:
    """Paired subject-level contrast plot, styled after H2 paired-change plots."""
    if not HAS_MPL:
        return False

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    columns = ["self_count", "own_label_preset_count", "mismatch_mean_count"]
    labels = ["Self", "Own-label", "Mismatch mean"]
    x = np.arange(len(columns))

    panels: list[tuple[str, pd.DataFrame]] = [("All subjects", contrasts)]
    panels.extend((label, contrasts[contrasts["following_label"] == label]) for label in LABEL_ORDER)

    max_count = float(contrasts[columns].max().max()) if not contrasts.empty else 0.0
    y_top = max(5.0, np.ceil((max_count + 2.0) / 2.0) * 2.0)
    point_colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, len(panels), figsize=(15.2, 5.2), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Subject-Level Paired Takeover Contrasts", fontsize=14, fontweight="bold")

    for ax, (panel_label, data) in zip(axes, panels):
        data = data.dropna(subset=columns)
        for _, row in data.iterrows():
            values = [row[col] for col in columns]
            ax.plot(x, values, color="#111111", alpha=0.35, linewidth=0.8)

        for idx, (column, color) in enumerate(zip(columns, point_colors)):
            ax.scatter(
                np.full(len(data), idx),
                data[column],
                color=color,
                edgecolors="#333333",
                linewidths=0.35,
                alpha=0.75,
                s=26,
                zorder=3,
            )

        means = [float(data[col].mean()) if len(data) else np.nan for col in columns]
        ax.plot(
            x,
            means,
            color="#111111",
            marker="D",
            markerfacecolor="white",
            markeredgecolor="#333333",
            linewidth=2.0,
        )
        for i, mean in enumerate(means):
            if pd.notna(mean):
                ax.text(i, min(mean + 0.35, y_top - 0.25), f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

        diff = means[0] - means[2] if pd.notna(means[0]) and pd.notna(means[2]) else np.nan
        title = f"{panel_label}\nSelf - mismatch={diff:+.2f}" if pd.notna(diff) else panel_label
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylim(-0.5, y_top)
        ax.set_yticks(np.arange(0, y_top + 1, 2))
        ax.grid(True, axis="y", linestyle="--", color="#cccccc", alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("L3 takeover count")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def first_row(df: pd.DataFrame, **filters: str) -> pd.Series | None:
    sub = df.copy()
    for key, value in filters.items():
        sub = sub[sub[key] == value]
    if sub.empty:
        return None
    return sub.iloc[0]


def markdownish_table(df: pd.DataFrame) -> str:
    """Return a plain table that renders cleanly in Markdown without tabulate."""
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_summary(
    detail: pd.DataFrame,
    group_style: pd.DataFrame,
    contrasts: pd.DataFrame,
    contrast_test_table: pd.DataFrame,
    friedman_table: pd.DataFrame,
    mixed_table: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# L3 takeover count by Following Label and trajectory style")
    lines.append("")
    lines.append(f"- Subjects: {detail['subject'].nunique()}")
    lines.append(f"- L3 style observations: {len(detail)}")
    lines.append("- Lower takeover count means fewer manual interventions.")
    lines.append("- One-sided Wilcoxon tests use alternative: left condition < right condition.")
    lines.append(
        "- Figures: l3_takeover_label_group_style_boxplots.png; "
        "l3_takeover_self_own_mismatch_paired_changes.png "
        "(also saved as l3_takeover_self_own_mismatch_lines.png)."
    )
    lines.append("")

    lines.append("## Label group sizes")
    for label in LABEL_ORDER:
        n = detail.loc[detail["following_label"] == label, "subject"].nunique()
        lines.append(f"- {label}: n={n}")
    lines.append("")

    lines.append("## Mean takeover counts by label group")
    pivot = group_style.pivot(index="following_label", columns="vehicle_style", values="mean").reindex(
        index=LABEL_ORDER,
        columns=[STYLE_LABELS[style] for style in STYLE_ORDER],
    )
    lines.append(markdownish_table(pivot.round(3).reset_index()))
    lines.append("")

    lines.append("## Overall mixed-design checks")
    lines.append(markdownish_table(mixed_table))
    lines.append("")
    lines.append("## Friedman style tests")
    lines.append(markdownish_table(friedman_table))
    lines.append("")

    lines.append("## Key aligned-vs-mismatch contrasts")
    for comparison in [
        "Self - mismatch preset mean",
        "Own-label preset - mismatch preset mean",
        "Mean(Self, own-label preset) - mismatch preset mean",
    ]:
        row = first_row(contrast_test_table, scope="All subjects", comparison=comparison)
        if row is None:
            continue
        lines.append(
            f"- {comparison}: left mean={row['left_mean']:.3f}, right mean={row['right_mean']:.3f}, "
            f"diff={row['mean_diff_left_minus_right']:+.3f}, "
            f"Wilcoxon one-sided p={row['p_wilcoxon_less']:.4f}{p_stars(row['p_wilcoxon_less'])}, "
            f"sign-test p={row['p_sign_less']:.4f}; "
            f"left lower/equal/higher={int(row['n_left_lower'])}/{int(row['n_equal'])}/{int(row['n_left_higher'])}."
        )
    lines.append("")

    lines.append("## By-label key contrast: Self - mismatch preset mean")
    for label in LABEL_ORDER:
        row = first_row(contrast_test_table, scope=label, comparison="Self - mismatch preset mean")
        if row is None:
            continue
        lines.append(
            f"- {label}: diff={row['mean_diff_left_minus_right']:+.3f}, "
            f"p={row['p_wilcoxon_less']:.4f}, left lower/equal/higher="
            f"{int(row['n_left_lower'])}/{int(row['n_equal'])}/{int(row['n_left_higher'])}."
        )
    lines.append("")

    lines.append("## By-label key contrast: Own-label preset - mismatch preset mean")
    for label in LABEL_ORDER:
        row = first_row(contrast_test_table, scope=label, comparison="Own-label preset - mismatch preset mean")
        if row is None:
            continue
        lines.append(
            f"- {label}: diff={row['mean_diff_left_minus_right']:+.3f}, "
            f"p={row['p_wilcoxon_less']:.4f}, left lower/equal/higher="
            f"{int(row['n_left_lower'])}/{int(row['n_equal'])}/{int(row['n_left_higher'])}."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Self trajectories show a lower mean takeover count than the two non-own preset trajectories, "
        "but the one-sided paired Wilcoxon result should be interpreted together with the sign test and "
        "the small subgroup sizes. Own-label preset trajectories do not show a reliable lower count "
        "than mismatched presets in this sample."
    )
    (OUTPUT_DIR / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail = attach_labels(load_takeover_detail(), load_style_labels())
    wide = make_wide(detail)
    contrasts = build_subject_contrasts(wide)
    group_style, contrast_desc = descriptives(detail, contrasts)
    contrast_test_table = contrast_tests(contrasts)
    pairwise_table = style_pairwise_tests(wide)
    friedman_table = friedman_tests(wide)
    mixed_table = mixed_design_tests(wide)

    detail.to_csv(OUTPUT_DIR / "l3_takeover_label_detail.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUTPUT_DIR / "l3_takeover_label_wide.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUTPUT_DIR / "l3_takeover_subject_contrasts.csv", index=False, encoding="utf-8-sig")
    group_style.to_csv(OUTPUT_DIR / "l3_takeover_label_group_style_descriptives.csv", index=False, encoding="utf-8-sig")
    contrast_desc.to_csv(OUTPUT_DIR / "l3_takeover_contrast_descriptives.csv", index=False, encoding="utf-8-sig")
    contrast_test_table.to_csv(OUTPUT_DIR / "l3_takeover_aligned_vs_mismatch_tests.csv", index=False, encoding="utf-8-sig")
    pairwise_table.to_csv(OUTPUT_DIR / "l3_takeover_style_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    friedman_table.to_csv(OUTPUT_DIR / "l3_takeover_friedman_tests.csv", index=False, encoding="utf-8-sig")
    mixed_table.to_csv(OUTPUT_DIR / "l3_takeover_mixed_design_permutation_tests.csv", index=False, encoding="utf-8-sig")

    plot_group_style_means(group_style, detail, OUTPUT_DIR / "l3_takeover_label_group_style_boxplots.png")
    plot_contrast_lines(contrasts, OUTPUT_DIR / "l3_takeover_self_own_mismatch_paired_changes.png")
    plot_contrast_lines(contrasts, OUTPUT_DIR / "l3_takeover_self_own_mismatch_lines.png")
    write_summary(detail, group_style, contrasts, contrast_test_table, friedman_table, mixed_table)

    print(f"subjects: {wide['subject'].nunique()}")
    print(f"observations: {len(detail)}")
    print(f"output: {OUTPUT_DIR}")
    print()
    print(mixed_table.to_string(index=False))
    print()
    key = contrast_test_table[
        (contrast_test_table["scope"] == "All subjects")
        & contrast_test_table["comparison"].isin(
            [
                "Self - mismatch preset mean",
                "Own-label preset - mismatch preset mean",
                "Mean(Self, own-label preset) - mismatch preset mean",
            ]
        )
    ]
    print(key.to_string(index=False))


def main() -> int:
    try:
        write_outputs()
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

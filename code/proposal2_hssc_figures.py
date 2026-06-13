#!/usr/bin/env python3
"""Generate HSSC-ready findings tables and figures from STM report outputs."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PERIOD_ORDER = [
    "Georgian",
    "EarlyVictorian",
    "LateVictorian",
    "Edwardian",
    "EarlyModernist",
]
# Concise, parallel (Adjective + Noun) labels used across ALL figures and
# tables. The "masculinity/manliness" suffix is dropped to save space and to
# avoid over-claiming; the figure captions state once that all candidate
# dimensions are masculine social formations.
DIMENSION_LABELS = {
    "gentlemanly_respectability": "Gentlemanly Status",
    "domestic_paternal": "Domestic Paternalism",
    "aristocratic_chivalric": "Aristocratic Chivalry",
    "professional_commercial_breadwinner": "Professional Breadwinner",
    "imperial_frontier_adventure": "Imperial Adventure",
    "moral_christian_manliness": "Christian Moralism",
    "sentimental_romantic": "Sentimental Manhood",
    "none": "Gothic Romance",
}
PALETTE = [
    "#4f7f52",
    "#b35d4d",
    "#2f7f92",
    "#c08a2b",
    "#7b6cae",
    "#6d6d6d",
    "#9a6f4f",
    "#536b8f",
]
PERIOD_SHORT = {
    "Georgian": "Georg.",
    "EarlyVictorian": "E.Vic",
    "LateVictorian": "L.Vic",
    "Edwardian": "Edw.",
    "EarlyModernist": "E.Mod",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", default="outputs/stm_report_k5_hssc")
    parser.add_argument("--search_csv", default="")
    parser.add_argument("--robustness_report_dir", default="")
    parser.add_argument("--output_dir", default="outputs/hssc_figures_k5_seven_concepts")
    return parser.parse_args()


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def topic_label(row: pd.Series) -> str:
    # Always use the concise, parallel dimension label (not the long
    # specific_topic_label) so every figure/table shares one naming scheme.
    dimension = DIMENSION_LABELS.get(str(row["candidate_dimension"]), str(row["candidate_dimension"]))
    return f"T{int(row['topic_id'])}: {dimension}"


def short_words(words: str, n: int = 8) -> str:
    parts = [part.strip() for part in str(words).split(",") if part.strip()]
    return ", ".join(parts[:n])


def read_topic_summary(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "topic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing topic summary: {path}")
    df = pd.read_csv(path)
    df["topic_id"] = pd.to_numeric(df["topic_id"]).astype(int)
    df["candidate_dimension"] = df["candidate_dimension"].fillna("none")
    if "specific_topic_label" not in df.columns:
        df["specific_topic_label"] = ""
    df["topic_label"] = df.apply(topic_label, axis=1)
    return df


def candidate_topics(topic_summary: pd.DataFrame) -> list[int]:
    candidates = topic_summary[topic_summary["candidate_dimension"] != "none"]
    return candidates["topic_id"].astype(int).tolist()


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    display = df.fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        cells = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_topic_summary_table(topic_summary: pd.DataFrame, output_dir: Path) -> None:
    table = topic_summary.copy()
    table = table[
        [
            "topic_id",
            "candidate_dimension",
            "specific_topic_label",
            "overlap_words",
            "overlap_count",
            "prob_words",
            "frex_words",
            "semantic_coherence",
            "exclusivity",
        ]
    ].copy()
    table["candidate_dimension"] = table["candidate_dimension"].map(
        lambda value: DIMENSION_LABELS.get(str(value), str(value))
    )
    table["prob_words"] = table["prob_words"].map(short_words)
    table["frex_words"] = table["frex_words"].map(short_words)
    table["semantic_coherence"] = table["semantic_coherence"].round(3)
    table["exclusivity"] = table["exclusivity"].round(3)
    table.to_csv(output_dir / "table1_topic_summary_for_manuscript.csv", index=False)
    write_markdown_table(table, output_dir / "table1_topic_summary_for_manuscript.md")


def build_period_change_table(report_dir: Path, topic_summary: pd.DataFrame, output_dir: Path) -> None:
    ranges = pd.read_csv(report_dir / "period_ranges.csv")
    table = ranges.merge(
        topic_summary[
            [
                "topic_id",
                "candidate_dimension",
                "specific_topic_label",
                "overlap_words",
                "prob_words",
                "frex_words",
            ]
        ],
        on="topic_id",
        how="left",
    )
    table["candidate_dimension"] = table["candidate_dimension"].map(
        lambda value: DIMENSION_LABELS.get(str(value), str(value))
    )
    table["prob_words"] = table["prob_words"].map(short_words)
    table["frex_words"] = table["frex_words"].map(short_words)
    for col in ["highest_prevalence", "lowest_prevalence", "range"]:
        table[col] = table[col].round(4)
    table.to_csv(output_dir / "table2_period_change_topics.csv", index=False)
    write_markdown_table(table, output_dir / "table2_period_change_topics.md")


def build_gender_table(report_dir: Path, topic_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    gender = pd.read_csv(report_dir / "author_gender_prevalence.csv")
    candidates = candidate_topics(topic_summary)
    gender = gender[gender["topic_id"].isin(candidates)].copy()
    wide = gender.pivot(index="topic_id", columns="author_gender", values="mean_prevalence").reset_index()
    counts = gender.pivot(index="topic_id", columns="author_gender", values="n_novels").reset_index()
    table = wide.merge(
        topic_summary[
            [
                "topic_id",
                "candidate_dimension",
                "specific_topic_label",
                "overlap_words",
                "prob_words",
                "frex_words",
            ]
        ],
        on="topic_id",
        how="left",
    )
    if "F" in table.columns and "M" in table.columns:
        table["female_minus_male"] = table["F"] - table["M"]
        table["absolute_difference"] = table["female_minus_male"].abs()
    for gender_col in ["F", "M", "female_minus_male", "absolute_difference"]:
        if gender_col in table.columns:
            table[gender_col] = table[gender_col].round(4)
    for gender_col in ["F", "M"]:
        if gender_col in counts.columns:
            table[f"n_{gender_col}_novels"] = counts[gender_col]
    table["candidate_dimension"] = table["candidate_dimension"].map(
        lambda value: DIMENSION_LABELS.get(str(value), str(value))
    )
    table["prob_words"] = table["prob_words"].map(short_words)
    table["frex_words"] = table["frex_words"].map(short_words)
    table = table.sort_values("absolute_difference", ascending=False)
    table.to_csv(output_dir / "table3_author_gender_candidate_topics.csv", index=False)
    write_markdown_table(table, output_dir / "table3_author_gender_candidate_topics.md")
    return table


def build_robustness_table(
    report_dir: Path,
    robustness_report_dir: Path,
    output_dir: Path,
) -> None:
    if not robustness_report_dir.exists():
        return
    main = read_topic_summary(report_dir)
    robust = read_topic_summary(robustness_report_dir)
    rows = []
    for model, df in [("K10 main", main), ("K12 robustness", robust)]:
        candidates = df[df["candidate_dimension"] != "none"]
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "model": model,
                    "topic_id": int(row["topic_id"]),
                    "candidate_dimension": DIMENSION_LABELS.get(
                        str(row["candidate_dimension"]), str(row["candidate_dimension"])
                    ),
                    "specific_topic_label": row.get("specific_topic_label", ""),
                    "overlap_words": row["overlap_words"],
                    "prob_words": short_words(row["prob_words"]),
                    "frex_words": short_words(row["frex_words"]),
                    "semantic_coherence": round(float(row["semantic_coherence"]), 3),
                    "exclusivity": round(float(row["exclusivity"]), 3),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table4_k10_k12_candidate_topic_overview.csv", index=False)
    write_markdown_table(table, output_dir / "table4_k10_k12_candidate_topic_overview.md")


def plot_candidate_topic_prevalence(
    report_dir: Path,
    topic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    candidates = candidate_topics(topic_summary)
    ci_path = report_dir / "period_prevalence_ci.csv"
    has_ci = ci_path.exists()
    df = pd.read_csv(ci_path if has_ci else report_dir / "period_prevalence.csv")
    df = df[df["topic_id"].isin(candidates)].copy()
    df["period"] = pd.Categorical(df["period"], PERIOD_ORDER, ordered=True)
    df = df.sort_values(["period", "topic_id"])
    labels = topic_summary.set_index("topic_id")["topic_label"].to_dict()

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for idx, topic_id in enumerate(candidates):
        part = df[df["topic_id"] == topic_id]
        color = PALETTE[idx % len(PALETTE)]
        x = part["period"].astype(str)
        ax.plot(
            x,
            part["mean_prevalence"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=labels[topic_id],
            color=color,
        )
        if has_ci:
            ax.fill_between(
                x,
                part["ci_lower"],
                part["ci_upper"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax.set_ylabel("Mean topic prevalence")
    ax.set_xlabel("Historical period")
    ax.set_title("Candidate masculinity-related topic prevalence by period")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, output_dir / "figure1_candidate_topic_prevalence")


def plot_period_small_multiples(
    report_dir: Path,
    topic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Faceted small multiples: one panel per candidate formation, shared axes,
    95% CI ribbon, ordered by net Georgian->EarlyModernist change (rising
    formations first). Far cleaner than overplotting seven lines on one axis."""
    candidates = candidate_topics(topic_summary)
    ci_path = report_dir / "period_prevalence_ci.csv"
    has_ci = ci_path.exists()
    df = pd.read_csv(ci_path if has_ci else report_dir / "period_prevalence.csv")
    df = df[df["topic_id"].isin(candidates)].copy()
    labels = topic_summary.set_index("topic_id")["topic_label"].to_dict()

    # Net change per topic (last period minus first).
    net = {}
    for t in candidates:
        s = df[df["topic_id"] == t].set_index("period").reindex(PERIOD_ORDER)
        net[t] = float(s["mean_prevalence"].iloc[-1] - s["mean_prevalence"].iloc[0])
    ordered = sorted(candidates, key=lambda t: net[t], reverse=True)

    ncol = 3
    nrow = math.ceil(len(ordered) / ncol)
    ymax = (df["ci_upper"].max() if has_ci else df["mean_prevalence"].max()) * 1.08
    x = list(range(len(PERIOD_ORDER)))
    xshort = [PERIOD_SHORT[p] for p in PERIOD_ORDER]

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * 2.8, nrow * 2.4), sharex=True, sharey=True
    )
    axes = axes.ravel()
    for i, t in enumerate(ordered):
        ax = axes[i]
        s = df[df["topic_id"] == t].set_index("period").reindex(PERIOD_ORDER)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, s["mean_prevalence"], marker="o", color=color, linewidth=2.2, markersize=4)
        if has_ci:
            ax.fill_between(x, s["ci_lower"], s["ci_upper"], color=color, alpha=0.18, linewidth=0)
        name = labels[t].split(": ", 1)[1]
        arrow = "▲" if net[t] >= 0 else "▼"
        ax.set_title(f"{labels[t].split(':')[0]}  {name}\n{arrow} {net[t]:+.2f}", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xshort, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="y", labelsize=7)
        style_axes(ax)
    for j in range(len(ordered), len(axes)):
        axes[j].set_visible(False)

    fig.supylabel("Mean topic prevalence (95% CI)", fontsize=10)
    fig.suptitle(
        "Historical trajectories of candidate masculine formations", fontsize=12
    )
    save(fig, output_dir / "figure1_period_small_multiples")


def plot_model_comparison(search_csv: Path, output_dir: Path) -> None:
    if not str(search_csv) or not search_csv.exists() or search_csv.is_dir():
        return
    df = pd.read_csv(search_csv)
    for col in ["K", "exclus", "semcoh", "heldout", "residual"]:
        df[col] = pd.to_numeric(df[col])

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True)
    metrics = [
        ("heldout", "Held-out likelihood", "#4f7f52"),
        ("residual", "Residual", "#b35d4d"),
        ("semcoh", "Semantic coherence", "#7b6cae"),
        ("exclus", "Exclusivity", "#c08a2b"),
    ]

    for ax, (column, title, color) in zip(axes.ravel(), metrics):
        ax.plot(df["K"], df[column], marker="o", linewidth=2, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Number of topics (K)")
        style_axes(ax)

    fig.suptitle("Model comparison across candidate topic numbers", fontsize=12)
    save(fig, output_dir / "figure2_model_comparison")


def plot_author_gender_candidates(
    report_dir: Path,
    topic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    candidates = candidate_topics(topic_summary)
    gender = pd.read_csv(report_dir / "author_gender_prevalence.csv")
    gender = gender[gender["topic_id"].isin(candidates)].copy()
    labels = topic_summary.set_index("topic_id")["topic_label"].to_dict()
    gender["topic_label"] = gender["topic_id"].map(labels)
    order = [labels[topic_id] for topic_id in candidates]

    pivot = (
        gender.pivot(index="topic_label", columns="author_gender", values="mean_prevalence")
        .reindex(order)
        .fillna(0)
    )

    # Asymmetric bootstrap 95% CI error bars (mean - lo, hi - mean) per gender.
    ci_path = report_dir / "author_gender_prevalence_ci.csv"
    err = {"F": None, "M": None}
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        ci["topic_label"] = ci["topic_id"].map(labels)
        for g in ("F", "M"):
            sub = ci[ci["author_gender"] == g].set_index("topic_label").reindex(order)
            lo = (sub["mean_prevalence"] - sub["ci_lower"]).clip(lower=0).to_numpy()
            hi = (sub["ci_upper"] - sub["mean_prevalence"]).clip(lower=0).to_numpy()
            err[g] = [lo, hi]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    y_positions = range(len(pivot))
    height = 0.36
    female = pivot["F"] if "F" in pivot.columns else pd.Series([0] * len(pivot), index=pivot.index)
    male = pivot["M"] if "M" in pivot.columns else pd.Series([0] * len(pivot), index=pivot.index)
    ebar = dict(ecolor="#444444", elinewidth=0.9, capsize=2.5)
    ax.barh([y - height / 2 for y in y_positions], female, height=height, label="Female authors",
            color="#7b6cae", xerr=err["F"], error_kw=ebar)
    ax.barh([y + height / 2 for y in y_positions], male, height=height, label="Male authors",
            color="#2f7f92", xerr=err["M"], error_kw=ebar)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([textwrap.fill(label, 30) for label in pivot.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean topic prevalence")
    ax.set_title("Candidate topic prevalence by author gender")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, output_dir / "figure3_author_gender_candidate_topics")


def plot_period_heatmap(
    report_dir: Path,
    topic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    candidates = candidate_topics(topic_summary)
    period = pd.read_csv(report_dir / "period_prevalence.csv")
    period = period[period["topic_id"].isin(candidates)].copy()
    labels = topic_summary.set_index("topic_id")["topic_label"].to_dict()
    period["topic_label"] = period["topic_id"].map(labels)
    period["period"] = pd.Categorical(period["period"], PERIOD_ORDER, ordered=True)
    matrix = period.pivot(index="topic_label", columns="period", values="mean_prevalence")
    matrix = matrix[PERIOD_ORDER]
    matrix = matrix.reindex([labels[topic_id] for topic_id in candidates])

    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.55 * len(matrix) + 1.5)))
    image = ax.imshow(matrix.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels(PERIOD_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([textwrap.fill(label, 30) for label in matrix.index], fontsize=8)
    ax.set_title("Candidate topic prevalence heatmap by period")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean prevalence")
    save(fig, output_dir / "figure4_candidate_topic_period_heatmap")


def plot_topic_quality(topic_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    candidates = topic_summary["candidate_dimension"] != "none"
    ax.scatter(
        topic_summary.loc[~candidates, "semantic_coherence"],
        topic_summary.loc[~candidates, "exclusivity"],
        s=55,
        color="#9a9a9a",
        label="Other topics",
    )
    ax.scatter(
        topic_summary.loc[candidates, "semantic_coherence"],
        topic_summary.loc[candidates, "exclusivity"],
        s=70,
        color="#b35d4d",
        label="Candidate topics",
    )
    for _, row in topic_summary.iterrows():
        ax.annotate(
            f"T{int(row['topic_id'])}",
            (row["semantic_coherence"], row["exclusivity"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Semantic coherence")
    ax.set_ylabel("Exclusivity")
    ax.set_title("Topic quality diagnostics")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, output_dir / "figure5_topic_quality")


def write_findings_index(output_dir: Path) -> None:
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    lines = ["# HSSC Findings Package", ""]
    lines.append("Generated tables and figures from the current STM report outputs.")
    lines.append("")
    for name in files:
        if name != "FINDINGS_INDEX.md":
            lines.append(f"- `{name}`")
    lines.append("")
    (output_dir / "FINDINGS_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    search_csv = Path(args.search_csv) if args.search_csv else Path("__missing__")
    robustness_report_dir = Path(args.robustness_report_dir) if args.robustness_report_dir else Path("__missing__")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_summary = read_topic_summary(report_dir)
    build_topic_summary_table(topic_summary, output_dir)
    build_period_change_table(report_dir, topic_summary, output_dir)
    build_gender_table(report_dir, topic_summary, output_dir)
    build_robustness_table(report_dir, robustness_report_dir, output_dir)
    plot_candidate_topic_prevalence(report_dir, topic_summary, output_dir)
    plot_period_small_multiples(report_dir, topic_summary, output_dir)
    plot_model_comparison(search_csv, output_dir)
    plot_author_gender_candidates(report_dir, topic_summary, output_dir)
    plot_period_heatmap(report_dir, topic_summary, output_dir)
    plot_topic_quality(topic_summary, output_dir)
    write_findings_index(output_dir)


if __name__ == "__main__":
    main()

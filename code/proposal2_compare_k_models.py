#!/usr/bin/env python3
"""Compare low-K STM report outputs by concept-family coverage and duplication."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REPORT_RE = re.compile(r"stm_report_k(\d+)_hssc$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--k_values", default="3,4,5,6,7,8")
    parser.add_argument("--output_dir", default="outputs/k_low_comparison")
    parser.add_argument("--report_prefix", default="stm_report_k")
    parser.add_argument("--report_suffix", default="_hssc")
    return parser.parse_args()


def split_words(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def compare_report(report_dir: Path, k: int) -> dict[str, object]:
    topic_summary = pd.read_csv(report_dir / "topic_summary.csv")
    candidates = topic_summary[topic_summary["candidate_dimension"] != "none"].copy()
    family_counts = candidates["candidate_dimension"].value_counts().sort_index()
    duplicated = family_counts[family_counts > 1]
    overlap_words = set()
    for value in candidates["overlap_words"]:
        overlap_words.update(split_words(value))

    mean_coherence = (
        float(candidates["semantic_coherence"].mean()) if not candidates.empty else None
    )
    mean_exclusivity = (
        float(candidates["exclusivity"].mean()) if not candidates.empty else None
    )

    return {
        "K": k,
        "topics": int(topic_summary["topic_id"].nunique()),
        "candidate_topics": int(len(candidates)),
        "families_detected": int(family_counts.shape[0]),
        "families": ";".join(family_counts.index.tolist()),
        "family_topic_counts": ";".join(
            f"{family}:{count}" for family, count in family_counts.items()
        ),
        "duplicated_families": ";".join(duplicated.index.tolist()),
        "duplicated_family_count": int(duplicated.shape[0]),
        "overlap_word_count": int(len(overlap_words)),
        "candidate_mean_semantic_coherence": mean_coherence,
        "candidate_mean_exclusivity": mean_exclusivity,
    }


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


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]

    rows = []
    for k in k_values:
        report_dir = outputs_dir / f"{args.report_prefix}{k}{args.report_suffix}"
        if not report_dir.exists():
            continue
        rows.append(compare_report(report_dir, k))

    if not rows:
        raise FileNotFoundError("No matching report folders found for requested K values")

    comparison = pd.DataFrame(rows).sort_values("K")
    for col in ["candidate_mean_semantic_coherence", "candidate_mean_exclusivity"]:
        comparison[col] = comparison[col].round(3)
    comparison.to_csv(output_dir / "k3_to_k8_concept_family_comparison.csv", index=False)
    write_markdown_table(
        comparison,
        output_dir / "k3_to_k8_concept_family_comparison.md",
    )

    # IMPORTANT (methodology): K must NOT be selected by how many a-priori
    # concept families it reproduces. Doing so makes "K=N recovers N families"
    # circular -- the taxonomy is then both the target and the selection rule.
    # K should be chosen on statistical quality (semantic coherence and
    # exclusivity) and interpretability/stability. The family-coverage columns
    # below are reported as a *descriptive consequence* of that choice, not as
    # the criterion for it.
    quality_ranked = comparison.sort_values(
        ["candidate_mean_semantic_coherence", "candidate_mean_exclusivity"],
        ascending=[False, False],
    )
    taxonomy_ranked = comparison.sort_values(
        ["families_detected", "duplicated_family_count", "candidate_topics"],
        ascending=[False, True, True],
    )
    summary = {
        "selection_warning": (
            "Do not select K by concept-family coverage; that is circular. "
            "Choose K on semantic coherence, exclusivity, and interpretability, "
            "then report family coverage descriptively."
        ),
        "best_by_quality_coherence_exclusivity": quality_ranked.iloc[0].to_dict(),
        "taxonomy_coverage_heuristic_descriptive_only": taxonomy_ranked.iloc[0].to_dict(),
    }
    (output_dir / "k_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"reports_compared": len(rows), "output_dir": str(output_dir)}, indent=2))
    print("WARNING: K should be chosen on coherence/exclusivity/interpretability, "
          "not on concept-family coverage (that selection is circular).")


if __name__ == "__main__":
    main()

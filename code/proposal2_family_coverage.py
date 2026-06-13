#!/usr/bin/env python3
"""Build a family-coverage table across K: which of the seven a-priori
masculinity families separate as their own STM topic, and at which K.

This makes explicit that the seven-family TAXONOMY is retained in full, while
the NUMBER of families that separate as distinct topics depends on K. A family
that does not appear at the main K is fused with a neighbouring topic, not
absent (e.g. Christian Moralism is fused with Domestic Paternalism at K=7 and
only separates at K=9).

Reads the per-K report folders produced by proposal2_stm_report.py
(topic_summary.csv) and writes table0_family_coverage.{csv,md}.

Usage:
  python proposal2_family_coverage.py \
    --outputs_dir outputs \
    --k_values 5,6,7,8,9 \
    --report_prefix stm_report_full_1930_k \
    --report_suffix _seven_concepts \
    --main_k 7 \
    --output_dir outputs/hssc_figures_full_1930_k7_seven_concepts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from proposal2_stm_report import CONCEPT_FAMILIES
from proposal2_hssc_figures import DIMENSION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--k_values", default="5,6,7,8,9")
    parser.add_argument("--report_prefix", default="stm_report_full_1930_k")
    parser.add_argument("--report_suffix", default="_seven_concepts")
    parser.add_argument("--main_k", type=int, default=7)
    parser.add_argument(
        "--output_dir", default="outputs/hssc_figures_full_1930_k7_seven_concepts"
    )
    return parser.parse_args()


def family_counts_by_k(
    outputs_dir: Path, k_values: list[int], prefix: str, suffix: str
) -> dict[int, pd.Series]:
    """For each available K, count how many topics map to each family."""
    counts: dict[int, pd.Series] = {}
    for k in k_values:
        path = outputs_dir / f"{prefix}{k}{suffix}" / "topic_summary.csv"
        if not path.exists():
            continue
        ts = pd.read_csv(path)
        counts[k] = ts["candidate_dimension"].value_counts()
    return counts


def write_markdown(df: pd.DataFrame, path: Path, footnote: str) -> None:
    display = df.fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        cells = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(footnote)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = sorted(int(v) for v in args.k_values.split(",") if v.strip())

    counts = family_counts_by_k(
        outputs_dir, k_values, args.report_prefix, args.report_suffix
    )
    if not counts:
        raise FileNotFoundError("No per-K topic_summary.csv reports found.")
    available_k = sorted(counts.keys())
    max_k = max(available_k)

    rows = []
    for order, family in enumerate(CONCEPT_FAMILIES, start=1):
        label = DIMENSION_LABELS.get(family, family)
        keywords = ", ".join(sorted(CONCEPT_FAMILIES[family])[:6])
        # First K at which the family appears as a topic.
        present_ks = [k for k in available_k if counts[k].get(family, 0) >= 1]
        first_k = present_ks[0] if present_ks else None
        n_at_main = int(counts.get(args.main_k, pd.Series(dtype=int)).get(family, 0))

        if n_at_main >= 1:
            status = f"Separate topic at K={args.main_k}"
            if n_at_main > 1:
                status += f" ({n_at_main} topics)"
        elif first_k is not None:
            status = f"Fused at K={args.main_k}; separates at K={first_k}"
        else:
            status = f"Not separated for K<= {max_k} (fused)"

        rows.append(
            {
                "order": order,
                "family": family,
                "label": label,
                "example_keywords": keywords,
                f"separate_at_K{args.main_k}": "yes" if n_at_main >= 1 else "no",
                "first_recovered_K": first_k if first_k is not None else "—",
                "status": status,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table0_family_coverage.csv", index=False)
    footnote = (
        f"*All seven families are retained as the a-priori theoretical lens. "
        f"\"Separate topic\" means the family was recovered as a distinct STM topic "
        f"at the stated K; a family that is not separated at K={args.main_k} is fused "
        f"with a neighbouring formation, not absent. The Gothic Romance topic "
        f"(probable words: door, night, emily, chamber, castle, convent) is a "
        f"non-masculine contextual topic and is therefore not listed as a family. "
        f"K values scanned: {', '.join(map(str, available_k))}.*"
    )
    write_markdown(df, output_dir / "table0_family_coverage.md", footnote)
    print(df.to_string(index=False))
    print(f"\nWrote table0_family_coverage.{{csv,md}} to {output_dir}")


if __name__ == "__main__":
    main()

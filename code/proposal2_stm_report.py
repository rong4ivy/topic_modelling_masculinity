"""
Summarize STM outputs into paper-facing CSV tables.

The report is intentionally descriptive. It does not automatically declare
that a topic "is masculinity"; it only flags candidate dimensions based on
top-word overlap so the paper can use conservative language.

Usage:
  python proposal2_stm_report.py --run_dir stm_results_k12_nb25 --output_dir stm_report_k12_nb25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PERIOD_ORDER = [
    "Georgian",
    "EarlyVictorian",
    "LateVictorian",
    "Edwardian",
    "EarlyModernist",
]

CONCEPT_FAMILIES = {
    "domestic_paternal": {
        "father",
        "mother",
        "husband",
        "wife",
        "home",
        "family",
        "child",
        "children",
        "son",
        "daughter",
        "household",
        "care",
        "marriage",
        "married",
        "parent",
        "boy",
        "girl",
    },
    "gentlemanly_respectability": {
        "gentleman",
        "gentlemen",
        "lady",
        "sir",
        "conduct",
        "character",
        "propriety",
        "reputation",
        "dignity",
        "respectable",
        "honour",
        "honor",
        "society",
        "rank",
        "respect",
        "address",
        "polite",
        "courtesy",
        "ladyship",
    },
    "aristocratic_chivalric": {
        "lord",
        "prince",
        "duke",
        "king",
        "queen",
        "knight",
        "master",
        "sword",
        "horse",
        "castle",
        "noble",
        "royal",
        "chivalry",
        "hero",
        "baron",
        "highness",
        "court",
        "crown",
        "honour",
        "honor",
    },
    "professional_commercial_breadwinner": {
        "business",
        "money",
        "office",
        "trade",
        "profession",
        "clerk",
        "bank",
        "work",
        "labour",
        "labor",
        "merchant",
        "employment",
        "income",
        "dollar",
        "pound",
        "debt",
        "credit",
        "property",
        "fortune",
        "capital",
    },
    "imperial_frontier_adventure": {
        "empire",
        "colony",
        "colonial",
        "frontier",
        "jungle",
        "sea",
        "ship",
        "captain",
        "soldier",
        "officer",
        "army",
        "war",
        "battle",
        "danger",
        "courage",
        "endurance",
        "wilderness",
        "forest",
        "island",
        "voyage",
        "travel",
        "enemy",
        "fight",
        "rifle",
        "camp",
        "scout",
        "native",
        "indian",
    },
    "moral_christian_manliness": {
        "god",
        "church",
        "christian",
        "moral",
        "virtue",
        "conscience",
        "duty",
        "prayer",
        "soul",
        "sin",
        "faith",
        "piety",
        "holy",
        "religion",
        "minister",
        "priest",
        "heaven",
        "evil",
        "goodness",
        "truth",
    },
    "sentimental_romantic": {
        "feeling",
        "love",
        "sympathy",
        "passion",
        "tender",
        "sorrow",
        "affection",
        "grief",
        "joy",
        "sensibility",
        "sentiment",
        "emotion",
        "melancholy",
        "misery",
        "happiness",
        "beloved",
        "dear",
    },
}


def read_topic_words(run_dir: Path, metric: str) -> pd.DataFrame:
    path = run_dir / f"topic_words_{metric}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing topic words: {path}")
    return pd.read_csv(path)


def top_words(df: pd.DataFrame, topic_id: int, n: int = 12) -> list[str]:
    return (
        df[df["topic_id"] == topic_id]
        .sort_values("rank")
        .head(n)["word"]
        .astype(str)
        .tolist()
    )


def candidate_dimension(words: list[str], min_overlap: int = 2) -> tuple[str, str, int]:
    word_set = set(words)
    scored = []
    for order, (label, keywords) in enumerate(CONCEPT_FAMILIES.items()):
        overlap = sorted(word_set & keywords)
        if len(overlap) >= min_overlap:
            scored.append((len(overlap), -order, label, overlap))
    if not scored:
        return "none", "", 0
    scored.sort(reverse=True)
    _, _, label, overlap = scored[0]
    return label, ";".join(overlap), len(overlap)


def specific_topic_label(candidate: str, words: list[str]) -> str:
    """Assign a readable manuscript label from top-word evidence."""
    word_set = set(words)
    if candidate == "domestic_paternal":
        if word_set & {"love", "feeling", "felt", "dear"}:
            return "domestic-paternal kinship and family feeling"
        return "domestic-paternal household relations"
    if candidate == "gentlemanly_respectability":
        return "gentlemanly respectability and social conduct"
    if candidate == "aristocratic_chivalric":
        return "aristocratic-chivalric status and romance"
    if candidate == "professional_commercial_breadwinner":
        if word_set & {"political", "system", "government", "class", "social"}:
            return "public order, class, and social systems"
        if word_set & {"money", "dollar", "trade", "clerk", "work"}:
            return "professional-commercial breadwinner activity"
        return "professional-commercial breadwinner masculinity"
    if candidate == "imperial_frontier_adventure":
        return "imperial-frontier adventure and martial movement"
    if candidate == "moral_christian_manliness":
        return "moral-Christian manliness"
    if candidate == "sentimental_romantic":
        return "sentimental-romantic masculinity"
    return "unlabeled topic"


def build_topic_summary(run_dir: Path) -> pd.DataFrame:
    prob = read_topic_words(run_dir, "prob")
    frex = read_topic_words(run_dir, "frex")
    topic_ids = sorted(prob["topic_id"].unique())
    rows = []
    for topic_id in topic_ids:
        prob_words = top_words(prob, topic_id, 12)
        frex_words = top_words(frex, topic_id, 12)
        label, overlap, overlap_count = candidate_dimension(prob_words + frex_words)
        rows.append(
            {
                "topic_id": topic_id,
                "candidate_dimension": label,
                "specific_topic_label": specific_topic_label(label, prob_words + frex_words),
                "overlap_words": overlap,
                "overlap_count": overlap_count,
                "prob_words": ", ".join(prob_words),
                "frex_words": ", ".join(frex_words),
            }
        )
    summary = pd.DataFrame(rows)
    quality_path = run_dir / "topic_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        if "topic_id" in quality.columns:
            summary = summary.merge(quality, on="topic_id", how="left")
    return summary


def build_novel_means(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "document_topic_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing document-topic matrix: {path}")
    theta = pd.read_csv(path)
    topic_cols = [col for col in theta.columns if col.startswith("topic_")]
    meta_aggs = {
        col: "first"
        for col in ["period", "author_gender", "year", "title"]
        if col in theta.columns
    }
    aggs = {col: "mean" for col in topic_cols}
    aggs.update(meta_aggs)
    return theta.groupby("novel_filename").agg(aggs).reset_index()


def prevalence_table(novel: pd.DataFrame, group_col: str) -> pd.DataFrame:
    topic_cols = [col for col in novel.columns if col.startswith("topic_")]
    rows = []
    for group, subset in novel.groupby(group_col, observed=False):
        for col in topic_cols:
            rows.append(
                {
                    group_col: group,
                    "topic_id": int(col.replace("topic_", "")),
                    "mean_prevalence": float(subset[col].mean()),
                    "std": float(subset[col].std(ddof=0)),
                    "n_novels": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_group_ci(
    novel: pd.DataFrame,
    group_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Novel-level bootstrap 95% CIs for mean topic prevalence per group.

    Resamples novels (not segments) within each group, so the uncertainty
    reflects the real number of independent novels rather than the inflated
    segment count. This is the descriptive counterpart to the model-based
    estimateEffect CIs exported by proposal2_run_stm.R.
    """
    topic_cols = [col for col in novel.columns if col.startswith("topic_")]
    rng = np.random.default_rng(seed)
    rows = []
    for group, subset in novel.groupby(group_col, observed=False):
        n = len(subset)
        if n == 0:
            continue
        for col in topic_cols:
            arr = subset[col].to_numpy(dtype=float)
            if n > 1:
                idx = rng.integers(0, n, size=(n_boot, n))
                boot_means = arr[idx].mean(axis=1)
                lo, hi = np.percentile(boot_means, [2.5, 97.5])
            else:
                lo = hi = float(arr.mean())
            rows.append(
                {
                    group_col: group,
                    "topic_id": int(col.replace("topic_", "")),
                    "mean_prevalence": float(arr.mean()),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "n_novels": int(n),
                }
            )
    return pd.DataFrame(rows)


def period_ranges(period_prev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic_id, subset in period_prev.groupby("topic_id"):
        subset = subset.sort_values("mean_prevalence", ascending=False)
        high = subset.iloc[0]
        low = subset.iloc[-1]
        rows.append(
            {
                "topic_id": int(topic_id),
                "highest_period": high["period"],
                "highest_prevalence": high["mean_prevalence"],
                "lowest_period": low["period"],
                "lowest_prevalence": low["mean_prevalence"],
                "range": high["mean_prevalence"] - low["mean_prevalence"],
            }
        )
    return pd.DataFrame(rows).sort_values("range", ascending=False)


def summarize_stm_run(run_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_summary = build_topic_summary(run_dir)
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False)

    novel = build_novel_means(run_dir)
    if "period" in novel.columns:
        novel["period"] = pd.Categorical(novel["period"], categories=PERIOD_ORDER, ordered=True)
    novel.to_csv(output_dir / "novel_topic_means.csv", index=False)

    period_prev = prevalence_table(novel, "period")
    period_prev.to_csv(output_dir / "period_prevalence.csv", index=False)
    period_ranges(period_prev).to_csv(output_dir / "period_ranges.csv", index=False)

    # Novel-level bootstrap CIs so figures/tables can show uncertainty instead
    # of bare point differences.
    bootstrap_group_ci(novel, "period").to_csv(
        output_dir / "period_prevalence_ci.csv", index=False
    )

    if "author_gender" in novel.columns:
        prevalence_table(novel, "author_gender").to_csv(
            output_dir / "author_gender_prevalence.csv", index=False
        )
        bootstrap_group_ci(novel, "author_gender").to_csv(
            output_dir / "author_gender_prevalence_ci.csv", index=False
        )

    summary = {
        "topics": int(topic_summary["topic_id"].nunique()),
        "candidate_topics": int((topic_summary["candidate_dimension"] != "none").sum()),
        "novels": int(novel["novel_filename"].nunique()),
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize STM run outputs")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = summarize_stm_run(args.run_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

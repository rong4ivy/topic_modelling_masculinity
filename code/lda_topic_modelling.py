"""
Proposal 2: Pilot Topic Model for Competing Models of Manhood
===============================================================
This is a PILOT implementation using gensim LDA. It is NOT STM.

What this does:
  1. Builds vocabulary + DTM from segments
  2. Fits LDA at multiple K values with multiple seeds (stability check)
  3. Presents ALL topics for manual inspection (not auto-labeled)
  4. Provides keyword-overlap as a HELPER, not the decision rule
  5. Aggregates topic proportions to NOVEL level before metadata comparison
  6. Treats correlation patterns as exploratory, not causal

What this does NOT do (and what STM would add):
  - Metadata does not enter the generative model
  - No principled uncertainty estimation on prevalence effects
  - No topic-content covariates (same topic, different words by period)
  - No compositional correction on correlation

For publication: validate with R stm package using segments.csv + corpus.txt

Usage:
    python proposal2_topics.py --segments_dir ./segments
    python proposal2_topics.py --segments_dir ./segments --k 20
"""

import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
from gensim.utils import simple_preprocess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# STOPWORDS
# ══════════════════════════════════════════════════════════════

STOPWORDS = set("""
a an the and or but in on at to for of is it i me my we he she him her his
they them their you your its this that these those was were be been am are
had has have do does did not no nor so as if than too very can will just
should now all any each few more most other some such only own same been
being from into through during before after above below between with about
against up down out off over under again further then once here there when
where why how what which who whom while both would could shall upon said one
much yet might though also ever even well quite say every know like little
great good old come made came went thing make way must never let first two
long back still away got see may us our thy thee thou shall ere hath doth
""".split())

# NOTE on mr/mrs/sir: kept OUT of stopwords because they may carry
# social meaning for a masculinity study. Test both ways.


# ══════════════════════════════════════════════════════════════
# MASCULINITY KEYWORD HELPER (secondary aid, NOT the decision rule)
# ══════════════════════════════════════════════════════════════
# Use these ONLY to flag topics for closer manual inspection.
# The final decision on which topics are "masculinity topics"
# must come from reading top words + representative passages.

KEYWORD_HELPER = {
    "domestic_paternal": {
        "father", "husband", "home", "family", "child", "children",
        "house", "household", "care", "wife", "son", "daughter",
    },
    "gentlemanly_social": {
        "gentleman", "honour", "character", "conduct", "respectable",
        "dignity", "manly", "propriety", "reputation", "society",
    },
    "professional_commercial": {
        "business", "money", "office", "trade", "profession", "clerk",
        "bank", "work", "labour", "merchant", "employment", "income",
    },
    "military_imperial": {
        "war", "soldier", "battle", "army", "courage", "enemy",
        "empire", "fight", "danger", "sword", "officer", "camp",
    },
    "moral_religious": {
        "god", "church", "christian", "moral", "virtue", "conscience",
        "duty", "prayer", "soul", "sin", "faith", "piety",
    },
    "emotional_sentimental": {
        "heart", "feeling", "love", "tears", "sympathy", "passion",
        "tender", "sorrow", "affection", "grief", "joy",
    },
}


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_segment(text, min_len=3, max_len=25):
    tokens = simple_preprocess(text, deacc=True, min_len=min_len, max_len=max_len)
    return [t for t in tokens if t not in STOPWORDS and len(t) >= min_len]


def build_corpus(corpus_file, no_below=10, no_above=0.7):
    logger.info("Tokenizing segments...")
    texts = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(preprocess_segment(line.strip()))

    logger.info(f"Building dictionary from {len(texts):,} documents...")
    dictionary = Dictionary(texts)
    logger.info(f"Raw vocabulary: {len(dictionary):,} words")

    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    logger.info(f"Filtered vocabulary: {len(dictionary):,} words")

    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, bow_corpus, texts


# ══════════════════════════════════════════════════════════════
# MODEL FITTING WITH MULTI-SEED STABILITY
# ══════════════════════════════════════════════════════════════

def fit_lda(dictionary, bow_corpus, texts, k, passes=15, seed=42):
    """Fit one LDA model and return model + coherence."""
    model = LdaMulticore(
        corpus=bow_corpus, id2word=dictionary,
        num_topics=k, passes=passes, workers=2,
        random_state=seed, per_word_topics=True,
    )
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence='c_v')
    return model, cm.get_coherence()


def search_k_with_seeds(dictionary, bow_corpus, texts, k_values,
                         passes=15, n_seeds=3):
    """
    Fit LDA at multiple K values, each with multiple seeds.
    Reports mean and std of coherence to assess stability.
    """
    results = []
    for k in k_values:
        coherences = []
        for seed in range(n_seeds):
            logger.info(f"Fitting K={k}, seed={seed}...")
            _, coh = fit_lda(dictionary, bow_corpus, texts, k, passes, seed=seed*42+7)
            coherences.append(coh)
        results.append({
            "K": k,
            "coherence_mean": float(np.mean(coherences)),
            "coherence_std": float(np.std(coherences)),
            "coherence_values": coherences,
        })
        logger.info(f"  K={k}: coherence={np.mean(coherences):.4f} +/- {np.std(coherences):.4f}")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# TOPIC INSPECTION
# ══════════════════════════════════════════════════════════════

def get_topic_words(model, dictionary, topn=20):
    topics = {}
    for tid in range(model.num_topics):
        word_probs = model.get_topic_terms(tid, topn=topn)
        topics[tid] = [(dictionary[wid], float(prob)) for wid, prob in word_probs]
    return topics


def keyword_overlap_helper(topic_words, keyword_sets, topn_to_check=20):
    """
    Flag topics that overlap with keyword sets. This is a HELPER
    for manual inspection, not the final labeling decision.
    """
    flags = {}
    for tid, words in topic_words.items():
        top_set = set(w for w, p in words[:topn_to_check])
        matches = {}
        for label, keywords in keyword_sets.items():
            overlap = top_set & keywords
            if overlap:
                matches[label] = sorted(overlap)
        if matches:
            flags[tid] = matches
    return flags


# ══════════════════════════════════════════════════════════════
# DOCUMENT-TOPIC MATRIX + NOVEL-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════

def get_doc_topic_matrix(model, bow_corpus):
    K = model.num_topics
    matrix = np.zeros((len(bow_corpus), K))
    for i, bow in enumerate(bow_corpus):
        for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
            matrix[i, tid] = prob
    return matrix


def aggregate_to_novel_level(doc_topic_matrix, meta_df):
    """
    Average topic proportions within each novel, then merge metadata.
    This prevents long novels from dominating period comparisons.
    
    Returns a DataFrame: one row per novel, columns = topic proportions + metadata.
    """
    K = doc_topic_matrix.shape[1]
    topic_cols = [f"topic_{i}" for i in range(K)]

    df = meta_df.copy()
    for i, col in enumerate(topic_cols):
        df[col] = doc_topic_matrix[:, i]

    # Metadata columns to keep (first value per novel)
    meta_cols = ["novel_id", "year", "period", "author", "author_gender", "title"]
    available_meta = [c for c in meta_cols if c in df.columns]

    # Average topic proportions per novel
    novel_topics = df.groupby("novel_filename")[topic_cols].mean()

    # Get novel-level metadata (first row per novel)
    novel_meta = df.groupby("novel_filename")[available_meta].first()

    # Merge (both indexed by novel_filename)
    novel_df = novel_topics.join(novel_meta).reset_index()

    return novel_df, topic_cols


def prevalence_by_group(novel_df, topic_cols, group_col):
    """
    Compute mean topic prevalence by group, at the NOVEL level.
    Each novel counts once regardless of length.
    """
    if group_col not in novel_df.columns:
        return pd.DataFrame()

    results = []
    for group, subset in novel_df.groupby(group_col):
        if len(subset) == 0:
            continue
        for col in topic_cols:
            tid = int(col.replace("topic_", ""))
            vals = subset[col].values
            results.append({
                group_col: group,
                "topic_id": tid,
                "mean_prevalence": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_novels": len(subset),
            })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# TOPIC ASSOCIATION PATTERNS (exploratory, not causal)
# ══════════════════════════════════════════════════════════════

def topic_associations(novel_df, topic_cols):
    """
    Compute correlation between topic proportions at the NOVEL level.
    
    WARNING: topic proportions are compositional (sum to ~1 per document).
    Negative correlations may be artifacts of the constraint, not
    evidence of competition. Treat these as exploratory patterns only.
    """
    matrix = novel_df[topic_cols].values
    corr = np.corrcoef(matrix.T)
    labels = [col.replace("topic_", "T") for col in topic_cols]
    return pd.DataFrame(corr, index=labels, columns=labels)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pilot LDA topic model for competing masculinities")
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--search_k_values", type=str, default="10,15,20,25")
    parser.add_argument("--passes", type=int, default=15)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--no_below", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir) if args.output_dir else segments_dir.parent / "results" / "topics"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(segments_dir / "segments.csv")
    # Exclude unknown periods
    n_unknown = (meta_df["period"] == "Unknown").sum()
    if n_unknown > 0:
        print(f"  Excluding {n_unknown} segments with Unknown period")
        meta_df = meta_df[meta_df["period"] != "Unknown"].reset_index(drop=True)
    print(f"Loaded {len(meta_df):,} segments from {meta_df['novel_filename'].nunique()} novels")

    # ── BUILD CORPUS ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING CORPUS")
    print("=" * 70)

    dictionary, bow_corpus, texts = build_corpus(
        segments_dir / "corpus.txt", no_below=args.no_below,
    )
    # Align with filtered metadata (if we dropped Unknown rows)
    # Re-read corpus lines to ensure alignment
    all_lines = (segments_dir / "corpus.txt").read_text(encoding="utf-8").strip().split("\n")
    # Keep only lines corresponding to non-Unknown segments
    # (segments.csv and corpus.txt are line-aligned)
    meta_full = pd.read_csv(segments_dir / "segments.csv")
    keep_mask = meta_full["period"] != "Unknown"
    kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
    if len(kept_indices) < len(all_lines):
        texts_filtered = [preprocess_segment(all_lines[i]) for i in kept_indices]
        bow_corpus = [dictionary.doc2bow(t) for t in texts_filtered]
        texts = texts_filtered

    print(f"  Documents: {len(bow_corpus):,}")
    print(f"  Vocabulary: {len(dictionary):,}")

    # ── MODEL SELECTION ───────────────────────────────────────
    if args.k is None:
        print("\n" + "=" * 70)
        print(f"MODEL SELECTION (searchK with {args.n_seeds} seeds per K)")
        print("=" * 70)

        k_values = [int(x) for x in args.search_k_values.split(",")]
        search_df = search_k_with_seeds(
            dictionary, bow_corpus, texts, k_values,
            passes=args.passes, n_seeds=args.n_seeds,
        )
        search_df.to_csv(output_dir / "search_k_results.csv", index=False)
        print(f"\n  Results:")
        for _, row in search_df.iterrows():
            print(f"    K={int(row['K']):3d}: coherence={row['coherence_mean']:.4f} "
                  f"+/- {row['coherence_std']:.4f}")

        best_k = int(search_df.loc[search_df["coherence_mean"].idxmax(), "K"])
        print(f"\n  Highest mean coherence at K={best_k}")
        print(f"  BUT: inspect all K values for interpretability.")
        print(f"  Override with --k N if a different K is more interpretable.")
    else:
        best_k = args.k
        print(f"\n  Using specified K={best_k}")

    # ── FIT FINAL MODEL ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FITTING FINAL MODEL (K={best_k})")
    print("=" * 70)

    model, final_coherence = fit_lda(
        dictionary, bow_corpus, texts, best_k, args.passes, args.seed)
    print(f"  Coherence: {final_coherence:.4f}")

    model.save(str(output_dir / "lda_model"))
    dictionary.save(str(output_dir / "dictionary"))

    # ── ALL TOPICS FOR MANUAL INSPECTION ──────────────────────
    print("\n" + "=" * 70)
    print("ALL TOPICS (inspect manually to identify masculinity topics)")
    print("=" * 70)

    topic_words = get_topic_words(model, dictionary, topn=20)
    topic_rows = []

    for tid in range(best_k):
        words = topic_words[tid]
        top10 = ", ".join(w for w, p in words[:10])
        print(f"\n  Topic {tid:2d}: {top10}")
        for w, p in words:
            topic_rows.append({"topic_id": tid, "word": w, "probability": p})

    pd.DataFrame(topic_rows).to_csv(output_dir / "topic_words.csv", index=False)

    # ── KEYWORD HELPER (secondary aid) ────────────────────────
    print("\n" + "=" * 70)
    print("KEYWORD OVERLAP HELPER (use as clue, not as final label)")
    print("=" * 70)

    flags = keyword_overlap_helper(topic_words, KEYWORD_HELPER)
    if flags:
        for tid, matches in flags.items():
            top5 = ", ".join(w for w, p in topic_words[tid][:5])
            print(f"\n  Topic {tid} ({top5}):")
            for label, overlap in matches.items():
                print(f"    ~ may relate to '{label}': {', '.join(overlap)}")
    else:
        print("  No topics matched keyword sets. This is normal.")
        print("  Read topic words above and decide manually which relate to manhood.")

    with open(output_dir / "keyword_flags.json", "w") as f:
        json.dump({str(k): v for k, v in flags.items()}, f, indent=2)
    # Also save as masculinity_topics.json for visualization compatibility
    with open(output_dir / "masculinity_topics.json", "w") as f:
        json.dump({str(k): v for k, v in flags.items()}, f, indent=2)

    print(f"\n  ACTION REQUIRED: Inspect topics above and decide which are")
    print(f"  masculinity-related. Then use those topic IDs in the visualization.")

    # ── DOCUMENT-TOPIC MATRIX ─────────────────────────────────
    print("\n" + "=" * 70)
    print("DOCUMENT-TOPIC PROPORTIONS")
    print("=" * 70)

    doc_topic = get_doc_topic_matrix(model, bow_corpus)
    np.save(output_dir / "doc_topic_matrix.npy", doc_topic)
    print(f"  Segment-level matrix: {doc_topic.shape}")

    # ── NOVEL-LEVEL AGGREGATION ───────────────────────────────
    print("\n" + "=" * 70)
    print("NOVEL-LEVEL AGGREGATION")
    print("(Each novel counts once, regardless of length)")
    print("=" * 70)

    novel_df, topic_cols = aggregate_to_novel_level(doc_topic, meta_df)
    novel_df.to_csv(output_dir / "novel_topic_means.csv", index=False)
    print(f"  Novels: {len(novel_df)}")
    print(f"  Columns: {topic_cols[:5]}... ({len(topic_cols)} topics)")

    # ── PREVALENCE BY PERIOD (novel-level) ────────────────────
    print("\n" + "=" * 70)
    print("TOPIC PREVALENCE BY PERIOD (novel-level means)")
    print("=" * 70)

    prev_period = prevalence_by_group(novel_df, topic_cols, "period")
    prev_period.to_csv(output_dir / "prevalence_by_period.csv", index=False)

    if not prev_period.empty:
        pivot = prev_period.pivot_table(
            index="topic_id", columns="period", values="mean_prevalence")
        print(f"\n{pivot.round(4).to_string()}")

    # ── PREVALENCE BY AUTHOR GENDER (novel-level) ─────────────
    print("\n" + "=" * 70)
    print("TOPIC PREVALENCE BY AUTHOR GENDER (novel-level means)")
    print("=" * 70)

    prev_gender = prevalence_by_group(novel_df, topic_cols, "author_gender")
    prev_gender.to_csv(output_dir / "prevalence_by_gender.csv", index=False)

    if not prev_gender.empty:
        pivot_g = prev_gender.pivot_table(
            index="topic_id", columns="author_gender", values="mean_prevalence")
        print(f"\n{pivot_g.round(4).to_string()}")

    # ── TOPIC ASSOCIATION PATTERNS (exploratory) ──────────────
    print("\n" + "=" * 70)
    print("TOPIC ASSOCIATION PATTERNS (exploratory, novel-level)")
    print("WARNING: compositional data -- negative correlations may be artifacts")
    print("=" * 70)

    corr = topic_associations(novel_df, topic_cols)
    corr.to_csv(output_dir / "topic_associations.csv")

    # Show only flagged topic pairs
    flagged_ids = list(flags.keys())
    if len(flagged_ids) >= 2:
        flagged_cols = [f"T{t}" for t in flagged_ids]
        sub_corr = corr.loc[
            [c for c in flagged_cols if c in corr.index],
            [c for c in flagged_cols if c in corr.columns]
        ]
        if not sub_corr.empty:
            print(f"\n  Associations among keyword-flagged topics:")
            print(sub_corr.round(3).to_string())

    # ── REPRESENTATIVE PASSAGES ───────────────────────────────
    print("\n" + "=" * 70)
    print("REPRESENTATIVE PASSAGES (for close reading)")
    print("=" * 70)

    # Use raw corpus for readable excerpts
    raw_path = segments_dir / "corpus_raw.txt"
    if raw_path.exists():
        raw_lines = raw_path.read_text(encoding="utf-8").strip().split("\n")
        # Re-filter to match kept_indices
        if len(kept_indices) < len(raw_lines):
            raw_lines = [raw_lines[i] for i in kept_indices]
    else:
        raw_lines = (segments_dir / "corpus.txt").read_text(encoding="utf-8").strip().split("\n")
        if len(kept_indices) < len(raw_lines):
            raw_lines = [raw_lines[i] for i in kept_indices]

    passage_rows = []
    for tid in range(best_k):
        top_docs = np.argsort(doc_topic[:, tid])[-5:][::-1]
        top5_words = ", ".join(w for w, p in topic_words[tid][:5])
        print(f"\n  Topic {tid} ({top5_words}):")

        for doc_id in top_docs:
            prop = doc_topic[doc_id, tid]
            novel = meta_df.iloc[doc_id]["novel_filename"] if doc_id < len(meta_df) else "?"
            year = meta_df.iloc[doc_id]["year"] if doc_id < len(meta_df) else "?"
            pos = meta_df.iloc[doc_id]["position_in_novel"] if doc_id < len(meta_df) else "?"
            excerpt = raw_lines[doc_id][:250] if doc_id < len(raw_lines) else ""

            print(f"    [{prop:.3f}] {novel} ({year}, pos={pos})")
            print(f"           {excerpt[:150]}...")

            passage_rows.append({
                "topic_id": tid, "doc_id": doc_id,
                "proportion": prop, "novel": novel,
                "year": year, "position": pos,
            })

    pd.DataFrame(passage_rows).to_csv(output_dir / "representative_passages.csv", index=False)

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            print(f"  {f.name:<40s} {f.stat().st_size:>10,} bytes")

    print(f"\n  STATUS: This is a PILOT LDA model.")
    print(f"  For publication, validate with R stm package:")
    print(f"    library(stm)")
    print(f"    segments <- read.csv('{segments_dir}/segments.csv')")
    print(f"    # ... see STM vignette for full workflow")
    print(f"\n  Next steps:")
    print(f"    1. Inspect ALL topics above and decide which relate to manhood")
    print(f"    2. Record your chosen topic IDs")
    print(f"    3. Run visualization: python proposal2_visualize.py --topics_dir {output_dir}")


if __name__ == "__main__":
    main()

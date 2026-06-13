"""
Prepare STM-ready inputs for the masculinity topic-modeling paper.

This script deliberately leaves the existing gensim/LDA pilot untouched.
It creates a transparent, auditable corpus that an R STM script can read:

  stm_inputs/documents.csv
      One row per segment, with metadata and a cleaned text column.
  stm_inputs/vocab.txt
      One retained token per line.
  stm_inputs/counts.tsv
      Long-form document-term counts for auditing.
  stm_inputs/removed_document_specific_terms.txt
      Probable character/proper-name tokens removed by concentration.
  stm_inputs/prep_summary.json
      Corpus and vocabulary summary.

Usage:
  python proposal2_stm_prep.py --segments_dir ./segments --output_dir ./stm_inputs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


PERIOD_ORDER = [
    "Georgian",
    "EarlyVictorian",
    "LateVictorian",
    "Edwardian",
    "EarlyModernist",
]

MASCULINITY_KEEP_TERMS = {
    "address",
    "affection",
    "army",
    "bank",
    "baron",
    "battle",
    "beloved",
    "boy",
    "business",
    "camp",
    "capital",
    "captain",
    "care",
    "castle",
    "character",
    "child",
    "children",
    "chivalry",
    "christian",
    "church",
    "class",
    "clerk",
    "colonial",
    "colony",
    "conscience",
    "conduct",
    "courage",
    "courtesy",
    "court",
    "credit",
    "crown",
    "danger",
    "daughter",
    "dear",
    "debt",
    "dignity",
    "dollar",
    "duke",
    "duty",
    "emotion",
    "employment",
    "endurance",
    "enemy",
    "empire",
    "evil",
    "faith",
    "father",
    "feeling",
    "fight",
    "forest",
    "fortune",
    "frontier",
    "gentleman",
    "gentlemen",
    "girl",
    "god",
    "goodness",
    "grief",
    "happiness",
    "heaven",
    "hero",
    "highness",
    "holy",
    "home",
    "honor",
    "honour",
    "horse",
    "household",
    "husband",
    "income",
    "indian",
    "island",
    "joy",
    "jungle",
    "king",
    "knight",
    "lady",
    "ladyship",
    "labor",
    "labour",
    "lord",
    "love",
    "man",
    "manhood",
    "manly",
    "marriage",
    "married",
    "master",
    "melancholy",
    "men",
    "merchant",
    "minister",
    "misery",
    "money",
    "moral",
    "mother",
    "native",
    "noble",
    "office",
    "officer",
    "parent",
    "passion",
    "piety",
    "polite",
    "political",
    "prayer",
    "priest",
    "prince",
    "profession",
    "property",
    "propriety",
    "queen",
    "rank",
    "religion",
    "reputation",
    "respect",
    "respectability",
    "respectable",
    "rifle",
    "royal",
    "scout",
    "sea",
    "sensibility",
    "sentiment",
    "ship",
    "sin",
    "sir",
    "social",
    "society",
    "soldier",
    "son",
    "sorrow",
    "soul",
    "soldier",
    "sympathy",
    "system",
    "tender",
    "trade",
    "travel",
    "truth",
    "virtue",
    "voyage",
    "war",
    "wife",
    "wilderness",
    "work",
}

BASE_STOPWORDS = set(
    """
    a an the and or but in on at to for of is it i me my we he she him her his
    they them their you your its this that these those was were be been am are
    had has have do does did not no nor so as if than too very can will just
    should now all any each few more most other some such only own same being
    from into through during before after above below between with about against
    up down out off over under again further then once here there when where why
    how what which who whom while both would could upon said one much yet might
    though also ever even well quite say every know like little great good old
    come made came went thing make way must never let first two long back still
    away got get see may us our thy thee thou shall ere hath doth don didn isn
    wasn weren couldn wouldn shouldn
    always something anything nothing yes yeah no oh ah ha i'm i'll i'd i've
    you're you'd you've he's he'll he'd she's she'll she'd we're we'll we'd
    they're they'll they'd it's it'll that's that'll there's here's what's
    who's don't doesn't didn't can't cannot won't ain't shan't aren't isn't
    wasn't weren't couldn't wouldn't shouldn't haven't hasn't hadn't
    """.split()
)

TOKEN_RE = re.compile(r"[a-z][a-z']{1,28}")


def normalize_token(token: str) -> str:
    token = token.lower().strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    return token


def light_lemma(token: str) -> str:
    """Conservative lemmatization without external model dependencies."""
    irregular = {
        "men": "men",
        "gentlemen": "gentlemen",
        "children": "children",
        "wives": "wife",
        "fathers": "father",
        "husbands": "husband",
        "soldiers": "soldier",
        "officers": "officer",
        "clerks": "clerk",
        "merchants": "merchant",
        "heroes": "hero",
    }
    if token in irregular:
        return irregular[token]
    if token in MASCULINITY_KEEP_TERMS:
        return token
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess_for_stm(text: str, extra_stops: set[str] | None = None) -> list[str]:
    stops = BASE_STOPWORDS | (extra_stops or set())
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = normalize_token(raw)
        if token in stops and token not in MASCULINITY_KEEP_TERMS:
            continue
        token = light_lemma(token)
        if len(token) < 3:
            continue
        if token in stops and token not in MASCULINITY_KEEP_TERMS:
            continue
        tokens.append(token)
    return tokens


def detect_document_specific_terms(
    token_docs: list[list[str]],
    doc_ids: Iterable[str],
    max_doc_fraction: float = 0.10,
    min_peak_freq: int = 20,
) -> set[str]:
    """Remove terms highly concentrated in a small share of novels/segments."""
    doc_ids = list(doc_ids)
    total_docs = len(set(doc_ids))
    max_docs = max(1, int(total_docs * max_doc_fraction))
    term_doc_counts: dict[str, set[str]] = defaultdict(set)
    term_doc_freqs: dict[str, Counter[str]] = defaultdict(Counter)

    for tokens, doc_id in zip(token_docs, doc_ids):
        counts = Counter(tokens)
        for term, count in counts.items():
            term_doc_counts[term].add(doc_id)
            term_doc_freqs[term][doc_id] += count

    removed = set()
    for term, docs in term_doc_counts.items():
        if term in MASCULINITY_KEEP_TERMS:
            continue
        if len(docs) <= max_docs and max(term_doc_freqs[term].values()) >= min_peak_freq:
            removed.add(term)
    return removed


def filter_vocabulary(
    token_docs: list[list[str]],
    no_below: int,
    no_above: float,
) -> set[str]:
    n_docs = len(token_docs)
    doc_freq = Counter()
    for tokens in token_docs:
        doc_freq.update(set(tokens))

    vocab = {
        term
        for term, freq in doc_freq.items()
        if freq >= no_below and (freq / max(n_docs, 1)) <= no_above
    }
    vocab.update(term for term in MASCULINITY_KEEP_TERMS if doc_freq.get(term, 0) > 0)
    return vocab


def build_stm_inputs(
    segments_dir: str | Path,
    output_dir: str | Path,
    no_below: int = 10,
    no_above: float = 0.70,
    remove_document_specific: bool = True,
    max_doc_fraction: float = 0.10,
    min_peak_freq: int = 20,
) -> dict[str, int | float | str]:
    segments_dir = Path(segments_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = segments_dir / "segments.csv"
    corpus_path = segments_dir / "corpus.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    meta = pd.read_csv(meta_path)
    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    if len(meta) != len(corpus_lines):
        raise ValueError(
            f"segments.csv rows ({len(meta)}) do not match corpus lines ({len(corpus_lines)})"
        )

    keep = meta["period"].isin(PERIOD_ORDER)
    meta = meta.loc[keep].reset_index(drop=True).copy()
    corpus_lines = [line for line, is_keep in zip(corpus_lines, keep) if bool(is_keep)]

    token_docs = [preprocess_for_stm(line) for line in corpus_lines]
    doc_ids = meta["novel_filename"].astype(str).tolist()

    removed_terms: set[str] = set()
    if remove_document_specific:
        removed_terms = detect_document_specific_terms(
            token_docs,
            doc_ids,
            max_doc_fraction=max_doc_fraction,
            min_peak_freq=min_peak_freq,
        )
        token_docs = [
            [token for token in tokens if token not in removed_terms]
            for tokens in token_docs
        ]

    vocab = filter_vocabulary(token_docs, no_below=no_below, no_above=no_above)
    token_docs = [[token for token in tokens if token in vocab] for tokens in token_docs]
    doc_lengths = [len(tokens) for tokens in token_docs]
    nonempty = [i for i, length in enumerate(doc_lengths) if length > 0]

    meta = meta.iloc[nonempty].reset_index(drop=True).copy()
    token_docs = [token_docs[i] for i in nonempty]
    doc_lengths = [doc_lengths[i] for i in nonempty]

    meta["doc_id"] = [f"seg_{int(seg_id):05d}" for seg_id in meta["seg_id"]]
    meta["clean_text"] = [" ".join(tokens) for tokens in token_docs]
    meta["clean_token_count"] = doc_lengths
    meta["period"] = pd.Categorical(meta["period"], categories=PERIOD_ORDER, ordered=True)

    docs_cols = [
        "doc_id",
        "seg_id",
        "novel_filename",
        "year",
        "period",
        "author_gender",
        "author",
        "title",
        "position_fraction",
        "word_count",
        "clean_token_count",
        "clean_text",
    ]
    available_cols = [col for col in docs_cols if col in meta.columns]
    meta[available_cols].to_csv(output_dir / "documents.csv", index=False)

    final_vocab = sorted({token for tokens in token_docs for token in tokens})
    (output_dir / "vocab.txt").write_text("\n".join(final_vocab) + "\n", encoding="utf-8")

    with (output_dir / "counts.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["doc_id", "term", "count"])
        for doc_id, tokens in zip(meta["doc_id"], token_docs):
            for term, count in sorted(Counter(tokens).items()):
                writer.writerow([doc_id, term, count])

    removed_path = output_dir / "removed_document_specific_terms.txt"
    removed_path.write_text("\n".join(sorted(removed_terms)) + "\n", encoding="utf-8")

    summary = {
        "documents": len(token_docs),
        "novels": int(meta["novel_filename"].nunique()),
        "vocabulary": len(final_vocab),
        "tokens": int(sum(doc_lengths)),
        "mean_tokens_per_document": float(sum(doc_lengths) / max(len(doc_lengths), 1)),
        "removed_document_specific_terms": len(removed_terms),
        "no_below": no_below,
        "no_above": no_above,
    }
    (output_dir / "prep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STM-ready corpus inputs")
    parser.add_argument("--segments_dir", default="./segments")
    parser.add_argument("--output_dir", default="./stm_inputs")
    parser.add_argument("--no_below", type=int, default=10)
    parser.add_argument("--no_above", type=float, default=0.70)
    parser.add_argument("--keep_document_specific", action="store_true")
    parser.add_argument("--max_doc_fraction", type=float, default=0.10)
    parser.add_argument("--min_peak_freq", type=int, default=20)
    parser.add_argument(
        "--no_protect",
        action="store_true",
        help="Robustness check: disable the protected masculinity vocabulary so "
        "all terms are subject to the same frequency/document-specificity filters.",
    )
    args = parser.parse_args()

    if args.no_protect:
        MASCULINITY_KEEP_TERMS.clear()

    summary = build_stm_inputs(
        segments_dir=args.segments_dir,
        output_dir=args.output_dir,
        no_below=args.no_below,
        no_above=args.no_above,
        remove_document_specific=not args.keep_document_specific,
        max_doc_fraction=args.max_doc_fraction,
        min_peak_freq=args.min_peak_freq,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Proposal 2, Step 1: Segment Novels into Document Windows
==========================================================
Splits each novel into non-overlapping, sentence-aware windows of
~2000 words. Each segment inherits metadata from the parent novel.

Design choices based on reviewer feedback:
  - Non-overlapping: overlap creates dependence between segments,
    inflating apparent topical continuity. For topic-prevalence
    comparisons, segments must be independent.
  - Sentence-aware: accumulates full sentences until hitting the
    target word count, avoiding mid-sentence cuts.
  - Novel-balanced metadata: records novel_id so downstream analysis
    can aggregate to novel level before comparing periods.

Usage:
    python proposal2_segment.py --cleaned_dir ../cleaned_texts \
                                 --metadata ../corpus_metadata.csv \
                                 --output_dir outputs/segments_hssc \
                                 --target_words 2000 \
                                 --end_year 1930
"""

import re
import csv
import argparse
from pathlib import Path
from collections import Counter


def sentence_split_simple(text):
    """Split text into sentences using punctuation + uppercase heuristic."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def segment_by_sentences(text, target_words=2000, min_words=800):
    """
    Accumulate full sentences until reaching ~target_words.
    Non-overlapping. Each segment is a clean block of complete sentences.
    Final segment is merged into the previous one if too short.
    """
    sentences = sentence_split_simple(text)
    if not sentences:
        return []

    segments = []
    current_sentences = []
    current_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        current_sentences.append(sent)
        current_wc += sent_wc

        if current_wc >= target_words:
            segments.append(" ".join(current_sentences))
            current_sentences = []
            current_wc = 0

    # Handle remainder
    if current_sentences:
        remainder = " ".join(current_sentences)
        if current_wc < min_words and segments:
            # Too short: merge with previous segment
            segments[-1] = segments[-1] + " " + remainder
        else:
            segments.append(remainder)

    return segments


def one_line(text):
    """Make a segment safe for one-line corpus files."""
    return " ".join(text.split())


def assign_period(year):
    """Assign period label. Returns 'Unknown' for invalid years."""
    if not isinstance(year, int) or year < 1700 or year > 2000:
        return "Unknown"
    if year <= 1836:
        return "Georgian"
    elif year <= 1875:
        return "EarlyVictorian"
    elif year <= 1901:
        return "LateVictorian"
    elif year <= 1914:
        return "Edwardian"
    elif year <= 1930:
        return "EarlyModernist"
    else:
        return "Post1930"


def main():
    parser = argparse.ArgumentParser(
        description="Segment novels into sentence-aware windows for topic modeling")
    parser.add_argument("--cleaned_dir", required=True)
    parser.add_argument("--metadata", default="corpus_metadata.csv")
    parser.add_argument("--output_dir", default="./segments")
    parser.add_argument("--target_words", type=int, default=2000)
    parser.add_argument("--min_words", type=int, default=800)
    parser.add_argument("--start_year", type=int, default=1771)
    parser.add_argument("--end_year", type=int, default=1930)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = list(csv.DictReader(f))

    print(f"Loaded metadata for {len(metadata)} novels")
    print(f"Target window: ~{args.target_words} words (sentence-aware, non-overlapping)")

    all_segments = []
    segment_texts = []       # lowercased for topic modeling
    segment_texts_raw = []   # original case for close reading
    seg_id = 0
    novels_processed = 0

    for row in metadata:
        fname = row["filename"]
        fpath = Path(args.cleaned_dir) / fname
        if not fpath.exists():
            continue

        text = fpath.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        segments = segment_by_sentences(text, args.target_words, args.min_words)
        if not segments:
            continue

        # Parse year safely
        year_str = row.get("date", "")
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            year = None

        if year is None or year < args.start_year or year > args.end_year:
            continue

        period = assign_period(year)
        if period == "Unknown" or period == "Post1930":
            continue

        novels_processed += 1

        for i, seg_text in enumerate(segments):
            seg_id += 1
            seg_text_raw = one_line(seg_text)
            seg_text_model = seg_text_raw.lower()
            wc = len(seg_text_raw.split())
            seg_meta = {
                "seg_id": seg_id,
                "novel_filename": fname,
                "novel_id": row.get("id", ""),
                "year": year_str,
                "period": period,
                "author": row.get("author", ""),
                "author_gender": row.get("gender_code", row.get("gender", "")),
                "title": row.get("title", ""),
                "position_in_novel": i,
                "total_segments": len(segments),
                "position_fraction": round(i / max(len(segments) - 1, 1), 3),
                "word_count": wc,
            }

            all_segments.append(seg_meta)
            segment_texts.append(seg_text_model)
            segment_texts_raw.append(seg_text_raw)

    if not all_segments:
        print("ERROR: No segments produced. Check cleaned_dir and metadata paths.")
        return

    # Save metadata CSV
    fieldnames = list(all_segments[0].keys())
    with open(output_dir / "segments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_segments)

    # Save corpus (one segment per line, lowercased for topic modeling)
    with open(output_dir / "corpus.txt", "w", encoding="utf-8") as f:
        for text in segment_texts:
            f.write(text + "\n")

    # Save raw corpus (original case, for close reading / representative passages)
    with open(output_dir / "corpus_raw.txt", "w", encoding="utf-8") as f:
        for text in segment_texts_raw:
            f.write(text + "\n")

    # Report
    print(f"\n{'='*60}")
    print(f"SEGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Novels processed: {novels_processed}")
    print(f"  Year range:        {args.start_year}-{args.end_year}")
    print(f"  Total segments:   {len(all_segments):,}")
    print(f"  Avg words/seg:    {sum(s['word_count'] for s in all_segments) // len(all_segments):,}")
    wcs = [s["word_count"] for s in all_segments]
    print(f"  Min/Max words:    {min(wcs):,} / {max(wcs):,}")

    print(f"\n  By period:")
    period_counts = Counter(s["period"] for s in all_segments)
    for period in [
        "Georgian",
        "EarlyVictorian",
        "LateVictorian",
        "Edwardian",
        "EarlyModernist",
        "Unknown",
    ]:
        if period in period_counts:
            print(f"    {period}: {period_counts[period]:,} segments")

    print(f"\n  By author gender:")
    gender_counts = Counter(s["author_gender"] for s in all_segments)
    for g in sorted(gender_counts):
        print(f"    {g}: {gender_counts[g]:,} segments")

    # Segments per novel distribution
    from collections import defaultdict
    segs_per_novel = defaultdict(int)
    for s in all_segments:
        segs_per_novel[s["novel_filename"]] += 1
    counts = list(segs_per_novel.values())
    print(f"\n  Segments per novel: min={min(counts)}, max={max(counts)}, "
          f"median={sorted(counts)[len(counts)//2]}")

    print(f"\n  Output:")
    print(f"    {output_dir / 'segments.csv'}    (metadata)")
    print(f"    {output_dir / 'corpus.txt'}      (lowercased, for topic model)")
    print(f"    {output_dir / 'corpus_raw.txt'}  (original case, for close reading)")
    print(f"\n  Next: python proposal2_topics.py --segments_dir {output_dir}")


if __name__ == "__main__":
    main()

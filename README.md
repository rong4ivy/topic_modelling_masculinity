# Masculine Social Formations in English-Language Fiction (1771–1930)

Reproducible code and outputs for a structural topic modeling (STM) study of
masculine social formations in 150 English-language novels, accompanying the
paper *From Gentlemanly Status to Professional Breadwinner: Structural Topic
Modeling of Masculine Formations in English-Language Fiction, 1771–1930*.

The study treats masculinity as a set of recurrent thematic **formations** rather
than a fixed lexical category, fits an STM with period and author-gender metadata,
selects the number of topics on a substantive criterion, and quantifies
uncertainty by bootstrapping whole novels. A central methodological point: the
seven theory-driven "masculine families" are an **interpretive lens applied after
estimation** — they label topics, they do not drive the model.

## Repository layout

```
code/      analysis pipeline (Python + R)
paper/     manuscript (LaTeX) + bibliography + compiled PDF
data/      corpus metadata only (full texts are NOT redistributed here)
figures/   manuscript-ready figures (PNG/PDF) and tables (CSV/MD)
results/   key result tables for the main K=7 model
```

## Data availability

The corpus is the English-language subset of Andrew Piper's **txtLAB Novel450**
dataset. The full novel texts are **not** included here (size and redistribution
considerations); only metadata (`data/corpus_metadata.csv`,
`data/txtlab_Novel450.csv`) is provided. Obtain the texts from the open dataset:

- Piper, A. *txtLAB Multilingual Novels.* figshare. https://doi.org/10.6084/m9.figshare.2062002.v3 (2016).
- Project site: https://txtlab.org/novel450/

Place the cleaned `.txt` files in a `cleaned_texts/` directory whose filenames
match the `filename` column of `corpus_metadata.csv`.

## Pipeline

Run from the repository root (adjust `--cleaned_dir` to your local texts):

```bash
# 1. Segment novels into ~2000-word, sentence-aware windows
python code/proposal2_segment.py \
  --cleaned_dir cleaned_texts --metadata data/corpus_metadata.csv \
  --output_dir outputs/segments --target_words 2000 --min_words 800 \
  --start_year 1771 --end_year 1930

# 2. Build STM-ready inputs (vocabulary, document table, counts)
python code/proposal2_stm_prep.py \
  --segments_dir outputs/segments --output_dir outputs/stm_inputs \
  --no_below 25 --no_above 0.65
#   add --no_protect for the protected-vocabulary robustness check

# 3. Fit STM: searchK over K=3..15, final model at K=7, with effect estimates
Rscript code/proposal2_run_stm.R \
  --input_dir outputs/stm_inputs --output_dir outputs/stm_results \
  --k_values 3,4,5,6,7,8,9,10,11,12 --final_k 7 --max_em_its 75

# 4. Paper-facing tables + novel-level bootstrap CIs
python code/proposal2_stm_report.py \
  --run_dir outputs/stm_results --output_dir outputs/stm_report

# 5. (optional) family coverage across K, and K-model comparison
python code/proposal2_family_coverage.py --outputs_dir outputs \
  --k_values 5,6,7,8,9 --main_k 7 --output_dir outputs/figures
python code/proposal2_compare_k_models.py --outputs_dir outputs --k_values 5,6,7,8,9

# 6. Figures + manuscript tables
python code/proposal2_hssc_figures.py \
  --report_dir outputs/stm_report \
  --search_csv outputs/stm_results/search_k.csv \
  --output_dir outputs/figures
```

`code/lda_topic_modelling.py` is a gensim-LDA pilot retained for
triangulation; STM is the method used in the paper.

## Requirements

- Python 3.10+ — see `requirements.txt` (`pandas`, `numpy`, `matplotlib`,
  `gensim` for the LDA pilot).
- R 4.x with the `stm` package: `install.packages("stm")`.

## Building the paper

```bash
cd paper && pdflatex paper_hssc && bibtex paper_hssc && pdflatex paper_hssc && pdflatex paper_hssc
```

## License

Code is released under the MIT License (`LICENSE`). The underlying corpus is
governed by the terms of the txtLAB Novel450 dataset.

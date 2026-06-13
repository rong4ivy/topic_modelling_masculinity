#!/usr/bin/env Rscript

# Run Structural Topic Models for the proposal 2 publication pipeline.
#
# Inputs:
#   stm_inputs/documents.csv from proposal2_stm_prep.py
#
# Outputs:
#   stm_results/search_k.csv
#   stm_results/stm_model_k<K>.rds
#   stm_results/topic_words_*.csv
#   stm_results/topic_quality.csv
#   stm_results/document_topic_matrix.csv
#   stm_results/representative_segments.csv
#   stm_results/effects_period_authorgender.rds
#
# Example:
#   Rscript proposal2_run_stm.R --input_dir stm_inputs --output_dir stm_results \
#     --k_values 10,12,15 --final_k 12 --max_em_its 75

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    input_dir = "stm_inputs",
    output_dir = "stm_results",
    k_values = "8,10,12,15",
    final_k = NA_integer_,
    prevalence = "period + author_gender",
    seed = 42L,
    max_em_its = 75L,
    skip_search = FALSE
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--skip_search") {
      out$skip_search <- TRUE
      i <- i + 1L
      next
    }
    if (i == length(args)) {
      stop("Missing value for argument: ", key)
    }
    value <- args[[i + 1L]]
    if (key == "--input_dir") out$input_dir <- value
    else if (key == "--output_dir") out$output_dir <- value
    else if (key == "--k_values") out$k_values <- value
    else if (key == "--final_k") out$final_k <- as.integer(value)
    else if (key == "--prevalence") out$prevalence <- value
    else if (key == "--seed") out$seed <- as.integer(value)
    else if (key == "--max_em_its") out$max_em_its <- as.integer(value)
    else stop("Unknown argument: ", key)
    i <- i + 2L
  }
  out
}

require_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Required R package '", pkg, "' is not installed.\n",
      "Install it with: install.packages('", pkg, "')\n",
      call. = FALSE
    )
  }
}

write_topic_words <- function(model, out_dir, n = 20L) {
  labels <- stm::labelTopics(model, n = n)
  categories <- c("prob", "frex", "lift", "score")

  for (category in categories) {
    mat <- labels[[category]]
    rows <- list()
    idx <- 1L
    for (topic_id in seq_len(nrow(mat))) {
      for (rank in seq_len(ncol(mat))) {
        rows[[idx]] <- data.frame(
          topic_id = topic_id,
          rank = rank,
          word = mat[topic_id, rank],
          metric = category,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
    }
    utils::write.csv(
      do.call(rbind, rows),
      file.path(out_dir, paste0("topic_words_", category, ".csv")),
      row.names = FALSE
    )
  }
}

write_representative_segments <- function(model, meta, out_dir, top_n = 5L) {
  theta <- model$theta
  rows <- list()
  idx <- 1L
  for (topic_id in seq_len(ncol(theta))) {
    ranked <- order(theta[, topic_id], decreasing = TRUE)
    ranked <- ranked[seq_len(min(top_n, length(ranked)))]
    for (doc_index in ranked) {
      rows[[idx]] <- data.frame(
        topic_id = topic_id,
        topic_proportion = theta[doc_index, topic_id],
        doc_id = meta$doc_id[doc_index],
        novel_filename = meta$novel_filename[doc_index],
        year = meta$year[doc_index],
        period = meta$period[doc_index],
        author_gender = meta$author_gender[doc_index],
        title = meta$title[doc_index],
        clean_text = meta$clean_text[doc_index],
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  utils::write.csv(
    do.call(rbind, rows),
    file.path(out_dir, "representative_segments.csv"),
    row.names = FALSE
  )
}

write_effect_estimates <- function(effects, out_dir) {
  # Export estimateEffect into a tidy CSV (coefficient, SE, 95% CI, p) so the
  # model's propagated uncertainty is actually usable downstream. Without this
  # the only uncertainty STM provides is discarded inside the .rds object.
  s <- tryCatch(summary(effects), error = function(e) NULL)
  if (is.null(s) || is.null(s$tables)) {
    message("Could not summarize estimateEffect; skipping effect_estimates.csv")
    return(invisible(NULL))
  }
  rows <- list()
  idx <- 1L
  for (topic_id in seq_along(s$tables)) {
    tab <- s$tables[[topic_id]]
    if (is.null(tab) || nrow(tab) == 0L) next
    for (term in rownames(tab)) {
      est <- tab[term, 1]
      se <- tab[term, 2]
      rows[[idx]] <- data.frame(
        topic_id = topic_id,
        term = term,
        estimate = est,
        std_error = se,
        t_value = tab[term, 3],
        p_value = tab[term, 4],
        ci_lower = est - 1.96 * se,
        ci_upper = est + 1.96 * se,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  if (length(rows) == 0L) {
    return(invisible(NULL))
  }
  utils::write.csv(
    do.call(rbind, rows),
    file.path(out_dir, "effect_estimates.csv"),
    row.names = FALSE
  )
}

csv_safe <- function(df) {
  for (name in names(df)) {
    if (is.list(df[[name]])) {
      df[[name]] <- vapply(
        df[[name]],
        function(x) paste(unlist(x), collapse = ";"),
        character(1)
      )
    }
  }
  df
}

main <- function() {
  args <- parse_args()
  require_package("stm")

  set.seed(args$seed)
  input_dir <- normalizePath(args$input_dir, mustWork = TRUE)
  output_dir <- args$output_dir
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  docs_path <- file.path(input_dir, "documents.csv")
  if (!file.exists(docs_path)) {
    stop("Missing STM documents file: ", docs_path)
  }

  message("Reading: ", docs_path)
  meta <- utils::read.csv(docs_path, stringsAsFactors = FALSE)
  required_cols <- c("clean_text", "period", "author_gender")
  missing_cols <- setdiff(required_cols, names(meta))
  if (length(missing_cols) > 0L) {
    stop("documents.csv is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  meta$period <- factor(
    meta$period,
    levels = c("Georgian", "EarlyVictorian", "LateVictorian", "Edwardian", "EarlyModernist"),
    ordered = FALSE
  )
  meta$author_gender <- factor(meta$author_gender)

  # Drop documents with missing period or author gender so they do not enter
  # the prevalence model as spurious factor levels (e.g. "" or NA).
  keep_rows <- !is.na(meta$period) & !is.na(meta$author_gender) &
    nzchar(as.character(meta$author_gender))
  if (any(!keep_rows)) {
    message("Dropping ", sum(!keep_rows), " documents with missing period/author_gender")
    meta <- meta[keep_rows, , drop = FALSE]
  }
  meta$period <- droplevels(meta$period)
  meta$author_gender <- droplevels(meta$author_gender)

  message("Processing text for stm...")
  processed <- stm::textProcessor(
    documents = meta$clean_text,
    metadata = meta,
    lowercase = FALSE,
    removestopwords = FALSE,
    removenumbers = FALSE,
    removepunctuation = FALSE,
    wordLengths = c(3, Inf),
    stem = FALSE,
    verbose = FALSE
  )

  prepped <- stm::prepDocuments(
    processed$documents,
    processed$vocab,
    processed$meta,
    lower.thresh = 1,
    verbose = FALSE
  )
  documents <- prepped$documents
  vocab <- prepped$vocab
  meta <- prepped$meta

  utils::write.csv(
    data.frame(
      documents = length(documents),
      vocabulary = length(vocab),
      novels = length(unique(meta$novel_filename)),
      stringsAsFactors = FALSE
    ),
    file.path(output_dir, "stm_input_summary.csv"),
    row.names = FALSE
  )

  prevalence_formula <- stats::as.formula(paste("~", args$prevalence))
  k_values <- as.integer(strsplit(args$k_values, ",", fixed = TRUE)[[1]])

  if (!args$skip_search) {
    message("Running searchK for K = ", paste(k_values, collapse = ", "))
    search <- stm::searchK(
      documents = documents,
      vocab = vocab,
      K = k_values,
      prevalence = prevalence_formula,
      data = meta,
      init.type = "Spectral",
      max.em.its = args$max_em_its,
      verbose = FALSE
    )
    saveRDS(search, file.path(output_dir, "search_k.rds"))
    utils::write.csv(csv_safe(search$results), file.path(output_dir, "search_k.csv"), row.names = FALSE)
  }

  final_k <- args$final_k
  if (is.na(final_k)) {
    if (!args$skip_search && exists("search")) {
      quality_cols <- intersect(c("semcoh", "exclus"), names(search$results))
      if (length(quality_cols) == 2L) {
        semcoh <- as.numeric(search$results$semcoh)
        exclus <- as.numeric(search$results$exclus)
        if (anyNA(semcoh) || anyNA(exclus)) {
          final_k <- k_values[[1]]
        } else {
          score <- scale(semcoh)[, 1] + scale(exclus)[, 1]
          final_k <- search$results$K[which.max(score)]
        }
      } else {
        final_k <- k_values[[1]]
      }
    } else {
      final_k <- k_values[[1]]
    }
  }
  message("Fitting final STM with K = ", final_k)
  model <- stm::stm(
    documents = documents,
    vocab = vocab,
    K = final_k,
    prevalence = prevalence_formula,
    data = meta,
    init.type = "Spectral",
    max.em.its = args$max_em_its,
    verbose = FALSE
  )
  saveRDS(model, file.path(output_dir, paste0("stm_model_k", final_k, ".rds")))

  quality <- data.frame(
    topic_id = seq_len(final_k),
    semantic_coherence = stm::semanticCoherence(model, documents),
    exclusivity = stm::exclusivity(model),
    stringsAsFactors = FALSE
  )
  utils::write.csv(quality, file.path(output_dir, "topic_quality.csv"), row.names = FALSE)
  write_topic_words(model, output_dir, n = 20L)

  theta <- as.data.frame(model$theta)
  names(theta) <- paste0("topic_", seq_len(ncol(theta)))
  theta <- cbind(meta[, intersect(c("doc_id", "novel_filename", "year", "period", "author_gender", "title"), names(meta))], theta)
  utils::write.csv(theta, file.path(output_dir, "document_topic_matrix.csv"), row.names = FALSE)
  write_representative_segments(model, meta, output_dir, top_n = 5L)

  message("Estimating effects: topics ~ ", args$prevalence)
  effects <- stm::estimateEffect(
    formula = stats::as.formula(paste("1:", final_k, "~", args$prevalence)),
    stmobj = model,
    metadata = meta,
    uncertainty = "Global"
  )
  saveRDS(effects, file.path(output_dir, "effects_period_authorgender.rds"))
  write_effect_estimates(effects, output_dir)

  utils::write.csv(
    data.frame(
      final_k = final_k,
      documents = length(documents),
      vocabulary = length(vocab),
      max_em_its = args$max_em_its,
      prevalence = args$prevalence,
      stringsAsFactors = FALSE
    ),
    file.path(output_dir, "run_summary.csv"),
    row.names = FALSE
  )

  message("STM results written to: ", normalizePath(output_dir))
}

main()

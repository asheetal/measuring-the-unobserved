#!/usr/bin/Rscript
# 1_environment_check.R
# Checks what is installed and reports what is missing.
#
#   ./1_environment_check.R
#
# Scoring pre-coded evaluations is the required path. Scoring raw text needs
# more, and this script only recommends what to install for it.

Sys.setenv(TF_USE_LEGACY_KERAS = "1")

R_PKGS_CORE <- c("tidyverse", "keras", "reticulate")
R_PKGS_CODE <- c("janitor", "data.table", "doParallel", "foreach", "future",
                 "DT", "readxl", "ff", "tokenizers", "textdata")
PY_MODS_CORE <- c("tensorflow")
PY_MODS_CODE <- c("torch", "transformers")

HERE      <- dirname(normalizePath(sub("^--file=", "",
              grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
DICT_FILE <- file.path(HERE, "tokenizer_word_index.csv")
COLS_FILE <- file.path(HERE, "construct_columns.txt")

fail <- FALSE
note <- character()

cat("Required, for scoring pre-coded evaluations\n")

miss_r <- R_PKGS_CORE[!vapply(R_PKGS_CORE, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss_r)) {
  cat("  installing R packages: ", paste(miss_r, collapse = ", "), "\n", sep = "")
  install.packages(miss_r, repos = "https://cloud.r-project.org")
  miss_r <- miss_r[!vapply(miss_r, requireNamespace, logical(1), quietly = TRUE)]
}
if (length(miss_r)) {
  cat("  R packages still missing: ", paste(miss_r, collapse = ", "), "\n", sep = ""); fail <- TRUE
} else cat("  R packages          ok\n")

py <- NA_character_
if (requireNamespace("reticulate", quietly = TRUE)) {
  suppressPackageStartupMessages(library(reticulate))
  py <- tryCatch(py_config()$python, error = function(e) NA_character_)
  cat("  python              ", if (is.na(py)) "not found" else py, "\n", sep = "")

  if (!py_module_available("tensorflow")) {
    cat("  tensorflow          MISSING. Install it into that Python:\n")
    cat("                      pip install tensorflow\n")
    cat("                      If R picks the wrong Python, set RETICULATE_PYTHON in ~/.Renviron.\n")
    fail <- TRUE
  } else {
    tf <- import("tensorflow")$`__version__`
    cat("  tensorflow          ", tf, "\n", sep = "")
    if (utils::compareVersion(sub("^(\\d+\\.\\d+).*", "\\1", tf), "2.14") < 0)
      note <- c(note, "TensorFlow is older than 2.14 and may fail to load the trained network.")
    if (!py_module_available("tf_keras")) {
      cat("  tf_keras            MISSING. The network was saved with Keras 2. Install it:\n")
      cat("                      pip install tf_keras\n")
      fail <- TRUE
    } else cat("  tf_keras            ok\n")
  }
}

for (f in c(DICT_FILE, COLS_FILE)) {
  if (file.exists(f)) cat("  ", basename(f), "  ok\n", sep = "")
  else { cat("  ", basename(f), "  MISSING\n", sep = ""); fail <- TRUE }
}

cat("\nOptional, for scoring raw text\n")

miss_r2 <- R_PKGS_CODE[!vapply(R_PKGS_CODE, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss_r2)) {
  cat("  R packages          missing: ", paste(miss_r2, collapse = ", "), "\n", sep = "")
  cat("                      install.packages(c(", paste(sprintf('"%s"', miss_r2), collapse = ", "), "))\n", sep = "")
} else cat("  R packages          ok\n")

if (!is.na(py)) {
  miss_py2 <- PY_MODS_CODE[!vapply(PY_MODS_CODE, py_module_available, logical(1))]
  if (length(miss_py2)) {
    cat("  python modules      missing: ", paste(miss_py2, collapse = ", "), "\n", sep = "")
    cat("                      pip install ", paste(miss_py2, collapse = " "), "\n", sep = "")
    cat("                      Match the torch build to the local CUDA driver, or install the CPU build.\n")
  } else {
    cat("  python modules      ok\n")
    gpu <- isTRUE(tryCatch(import("torch")$cuda$is_available(), error = function(e) FALSE))
    cat("  gpu                 ", if (gpu) "yes" else "no, the classifiers will run on CPU and be slow", "\n", sep = "")
  }
}

if (length(note)) { cat("\nNotes\n"); for (n in note) cat("  ", n, "\n", sep = "") }

cat("\n")
if (fail) {
  cat("Install what is marked MISSING, then run this script again.\n")
  quit(status = 1)
}
cat("Ready. Next: ./2_download_assets.R\n")

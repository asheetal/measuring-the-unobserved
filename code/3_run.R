#!/usr/bin/Rscript
# 3_run.R
# Scores written evaluations and prints a probability for each of four groups.
#
#   ./3_run.R                     score two pre-coded evaluations
#   ./3_run.R --complete          score all twenty pre-coded
#   ./3_run.R --full              score two raw evaluations, running the classifiers
#   ./3_run.R --full --complete   score all twenty raw
#   ./3_run.R mine.csv            score your own CSV
#
# The two shipped files hold the same twenty evaluations. examples_coded.csv
# already carries the 125 named qualities, so it needs no GPU and no
# classifiers. examples.csv holds the raw text and goes through all thirty-two
# classifiers first.
#
# Both files carry male and female columns. The instrument derives those two
# inputs from the instructor's name, and the name is coded ahead of time so that
# no name appears in this repository. A user working from their own records can
# supply a professor_name column instead and let the script derive them.
#
# Run ./1_environment_check.R and ./2_download_assets.R first.

Sys.setenv(TF_USE_LEGACY_KERAS = "1")

args     <- commandArgs(trailingOnly = TRUE)
FULL     <- "--full" %in% args
COMPLETE <- "--complete" %in% args
args     <- setdiff(args, c("--full", "--complete"))
if (length(args) > 1) {
  cat("usage: ./3_run.R [--full] [--complete] [file.csv]\n")
  quit(status = 1)
}

HERE       <- dirname(normalizePath(sub("^--file=", "",
                grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
ARTIFACTS  <- Sys.getenv("MU_ARTIFACTS", file.path(dirname(HERE), "artifacts"))
MODEL_FILE <- file.path(ARTIFACTS, "model_output_2025-03-27.hdf5")
DICT_FILE  <- file.path(HERE, "tokenizer_word_index.csv")
COLS_FILE  <- file.path(HERE, "construct_columns.txt")
DEFAULT_IN <- file.path(dirname(HERE),
                        if (FULL) "examples.csv" else "examples_coded.csv")
INFILE     <- if (length(args)) args[1] else DEFAULT_IN
MAXLEN     <- 512L
CLASSES    <- c("Asian", "Black", "Hispanic", "White")

for (f in c(DICT_FILE, COLS_FILE)) if (!file.exists(f)) {
  cat("Missing ", f, ". Run ./1_environment_check.R\n", sep = ""); quit(status = 1)
}
if (!file.exists(MODEL_FILE)) {
  cat("Missing the trained network. Run ./2_download_assets.R\n"); quit(status = 1)
}

suppressPackageStartupMessages({library(tidyverse); library(keras); library(reticulate)})

if (!file.exists(INFILE)) {
  cat("Missing ", INFILE, "\n", sep = ""); quit(status = 1)
}
df <- read_csv(INFILE, show_col_types = FALSE)
if (!"review_text" %in% names(df)) stop("input CSV needs a review_text column")

if (!COMPLETE) {
  n <- nrow(df)
  df <- head(df, 2)
  message("Scoring the first two of ", n, " evaluations. Add --complete for all of them.")
}

wanted   <- readLines(COLS_FILE)
PRECODED <- all(setdiff(wanted, c("male", "female")) %in% names(df))

if (PRECODED) {
  message("Input already carries the named qualities. Skipping the classifiers.")
  coded <- df
} else {
  jd <- path.expand("~/.cache/huggingface/hub/models--Reggie--muppet-roberta-base-joke_detector")
  if (!dir.exists(jd)) {
    cat("The joke detector is not in the Hugging Face cache. Run ./2_download_assets.R --coder\n")
    quit(status = 1)
  }
  if (identical(Sys.getenv("HF_HUB_OFFLINE"), "1"))
    message("HF_HUB_OFFLINE is set. Any classifier not yet cached will fail.")
  if (!isTRUE(tryCatch(import("torch")$cuda$is_available(), error = function(e) FALSE)))
    message("No usable GPU. The classifiers run on CPU and will take minutes.")
  source(file.path(HERE, "text2code_functions.gpu.R"))
  coded <- my_add_all_codes(data.frame(text = df$review_text), tempfile(fileext = ".rds"))
}

if (all(c("male", "female") %in% names(df))) {
  coded$male <- df$male; coded$female <- df$female
} else if ("professor_name" %in% names(df)) {
  g <- add_gender(data.frame(text = df$professor_name))
  coded$male <- round(g$male); coded$female <- 1 - coded$male
} else {
  message("No professor_name column. Gender inputs set to zero.")
  coded$male <- 0; coded$female <- 0
}

missing <- setdiff(wanted, names(coded))
if (length(missing)) {
  message("filling with zero: ", paste(missing, collapse = ", "))
  coded[missing] <- 0
}
x.constructs <- as.matrix(coded[, wanted])

VOCAB  <- 512L   # the embedding accepts indices 1 to VOCAB - 1
dic    <- read_csv(DICT_FILE, show_col_types = FALSE)
dic    <- dic[dic$key < VOCAB, ]
lookup <- setNames(dic$key, dic$word)
to_seq <- function(txt) {
  w <- str_split(str_squish(str_replace_all(str_to_lower(txt), "[^a-z0-9' ]", " ")), " ")[[1]]
  k <- lookup[w]; as.integer(k[!is.na(k)])
}
x.text <- pad_sequences(lapply(df$review_text, to_seq), maxlen = MAXLEN)

source(file.path(HERE, "loss_functions.R"))
model <- load_model_hdf5(MODEL_FILE, custom_objects = list(F1 = F1))
p <- predict(model, list(x.text, x.constructs))
colnames(p) <- CLASSES

idx <- max.col(p, ties.method = "first")

out <- tibble(
  review    = str_trunc(str_squish(df$review_text), 40),
  predicted = CLASSES[idx]
)
print(as.data.frame(out), row.names = FALSE)

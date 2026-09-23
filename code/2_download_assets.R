#!/usr/bin/Rscript
# 2_download_assets.R
# Downloads the trained network, and with --coder the archived joke detector too.
#
#   ./2_download_assets.R           the trained network only
#   ./2_download_assets.R --coder   also the joke detector, for scoring raw text

args   <- commandArgs(trailingOnly = TRUE)
CODER  <- "--coder" %in% args

HERE       <- dirname(normalizePath(sub("^--file=", "",
                grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
ARTIFACTS  <- Sys.getenv("MU_ARTIFACTS", file.path(dirname(HERE), "artifacts"))
ARTICLE_ID <- "33973036"   # figshare, doi 10.6084/m9.figshare.33973036
MODEL_FILE <- file.path(ARTIFACTS, "model_output_2025-03-27.hdf5")
JOKE_TGZ   <- file.path(ARTIFACTS, "archived_coders_muppet-roberta-base-joke_detector.tar.gz")
HF_HUB     <- path.expand("~/.cache/huggingface/hub")
JOKE_DIR   <- file.path(HF_HUB, "models--Reggie--muppet-roberta-base-joke_detector")


# The tarball carries snapshots but not the refs folder that maps a revision
# name to a snapshot, and transformers refuses to load the model without it.
fix_refs <- function(model_dir) {
  refs <- file.path(model_dir, "refs")
  if (file.exists(file.path(refs, "main"))) return(invisible(TRUE))
  snaps <- list.dirs(file.path(model_dir, "snapshots"), recursive = FALSE)
  if (!length(snaps)) return(invisible(FALSE))
  weight <- vapply(snaps, function(d)
    sum(file.size(list.files(d, full.names = TRUE, all.files = TRUE, no.. = TRUE)), na.rm = TRUE),
    numeric(1))
  best <- basename(snaps[which.max(weight)])
  dir.create(refs, showWarnings = FALSE, recursive = TRUE)
  writeLines(best, file.path(refs, "main"), sep = "")
  message("wrote refs/main -> ", best)
  invisible(TRUE)
}

options(timeout = 36000)
dir.create(ARTIFACTS, showWarnings = FALSE, recursive = TRUE)

# figshare serves files by numeric id, so look the ids up by name once.
figshare_index <- function() {
  con <- url(sprintf("https://api.figshare.com/v2/articles/%s/files", ARTICLE_ID), open = "rb")
  on.exit(close(con))
  txt <- paste(readLines(con, warn = FALSE), collapse = "")
  names  <- regmatches(txt, gregexpr('"name":\\s*"[^"]+"', txt))[[1]]
  names  <- sub('^"name":\\s*"', "", sub('"$', "", names))
  urls   <- regmatches(txt, gregexpr('"download_url":\\s*"[^"]+"', txt))[[1]]
  urls   <- sub('^"download_url":\\s*"', "", sub('"$', "", urls))
  setNames(urls, names)
}

INDEX <- tryCatch(figshare_index(), error = function(e) character())
if (!length(INDEX)) {
  cat("Cannot reach figshare. Check the connection and rerun.\n")
  quit(status = 1)
}

get_file <- function(path, min_bytes) {
  if (file.exists(path) && file.size(path) < min_bytes) {
    message("Partial download found. Removing ", basename(path))
    unlink(path)
  }
  if (file.exists(path)) { message("present: ", basename(path)); return(TRUE) }
  message("downloading ", basename(path))
  src <- INDEX[[basename(path)]]
  if (is.null(src)) { message("not on figshare: ", basename(path)); return(FALSE) }
  ok <- tryCatch({ utils::download.file(src, path, mode = "wb"); TRUE },
                 error = function(e) FALSE)
  if (!ok || !file.exists(path) || file.size(path) < min_bytes) {
    message("download failed: ", basename(path))
    unlink(path)
    return(FALSE)
  }
  TRUE
}

if (!get_file(MODEL_FILE, 1e6)) {
  cat("Cannot continue without the trained network. Check the connection and rerun.\n")
  quit(status = 1)
}

if (CODER) {
  if (dir.exists(JOKE_DIR)) {
    message("joke detector already in the Hugging Face cache")
    fix_refs(JOKE_DIR)
  } else if (get_file(JOKE_TGZ, 1e8)) {
    message("unpacking the joke detector into ", HF_HUB)
    dir.create(HF_HUB, showWarnings = FALSE, recursive = TRUE)
    utils::untar(JOKE_TGZ, exdir = HF_HUB)
    if (!dir.exists(JOKE_DIR)) {
      cat("The archive unpacked to an unexpected place. Check ", HF_HUB, "\n", sep = "")
      quit(status = 1)
    }
    fix_refs(JOKE_DIR)
  } else {
    cat("The joke detector could not be downloaded. Raw text scoring will fail without it,\n")
    cat("because the model was withdrawn from Hugging Face.\n")
    quit(status = 1)
  }
  message("\nThe other thirty-one classifiers download themselves on first use, which needs\n",
          "a Hugging Face account. See the README.")
}

cat("\nAssets in ", ARTIFACTS, "\n", sep = "")
cat("Next: ./3_run.R <file.csv>\n")

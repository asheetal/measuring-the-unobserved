# fetch_artifacts.R
# Downloads trained model artifacts from Zenodo into ./artifacts/
# Replace RECORD_ID with the Zenodo record id once the deposit is published.

RECORD_ID <- "0000000"                       # concept DOI 10.5281/zenodo.0000000
BASE      <- sprintf("https://zenodo.org/records/%s/files/%%s?download=1", RECORD_ID)
DEST      <- "artifacts"

files <- c(
  "word2vec_white.42B.model",
  "word2vec_black.42B.model",
  "word2vec_hispanic.42B.model",
  "word2vec_asian.42B.model",
  "model_output_2025-03-27.hdf5"
)

# gensim writes sidecar arrays for large models; list any that exist on the server.
sidecars <- c(
  # "word2vec_white.42B.model.wv.vectors.npy",
)

sha256 <- c(
  # "model_output_2025-03-27.hdf5" = "<paste checksum from Zenodo>",
)

dir.create(DEST, showWarnings = FALSE)

get_one <- function(f) {
  out <- file.path(DEST, f)
  if (file.exists(out)) {
    message("skip (present): ", f)
    return(invisible(out))
  }
  message("downloading: ", f)
  utils::download.file(sprintf(BASE, utils::URLencode(f)), out, mode = "wb", quiet = FALSE)
  if (f %in% names(sha256)) {
    got <- digest::digest(out, algo = "sha256", file = TRUE)
    if (!identical(got, unname(sha256[f]))) stop("checksum mismatch: ", f)
  }
  invisible(out)
}

invisible(lapply(c(files, sidecars), get_one))
message("artifacts ready in ./", DEST)

# Dual-Lens Construct Coding for Evaluative Text

A reusable R pipeline that pairs race-conditioned contextual word embeddings with a
neural construct coder, to measure stereotypic language in text that reads as favorable.

## What is here

Code, configuration, and a small sample corpus. Trained model artifacts are too large
for git and live on Zenodo under DOI 10.5281/zenodo.0000000.

## Artifacts

Four word2vec models trained on the 42B corpus, conditioned on instructor race
(white, black, hispanic, asian), and one trained dual-lens network saved as
`model_output_2025-03-27.hdf5`.
Note the filenames say `word2vec` because the models are word2vec trained on the
Common Crawl 42B corpus, not GloVe vectors.

## Setup

```r
renv::restore()          # R packages
source("fetch_artifacts.R")   # pulls models from Zenodo into ./artifacts/
```

The Keras backend runs on Python TensorFlow. Install it with the pinned versions in
`requirements.txt` and point `reticulate` at that environment before loading the model.

## Scoring new text

```r
source("predict.R")      # reads data/sample.csv, writes scored output
```

`predict.R` takes a CSV with one raw evaluation per row and returns the 123 coded
constructs plus the predicted ethnicity from the dual-lens model.

## Reproducibility

R package versions are pinned in `renv.lock`. Python versions are pinned in
`requirements.txt`. Random seeds are fixed in `config.R`. The RateMyProfessor corpus
is not redistributed here; `data/sample.csv` allows the pipeline to run end to end.

## License

TBD

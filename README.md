# measuring-the-unobserved

An instrument that measures implicit bias in written evaluations. It reads one
evaluation and returns two things: a score for how strongly the wording points to
the group of the person described, and the named qualities in the writing that
produced that score.

## What is here

The Python and R scripts that fine-tune the word embeddings, score the named
qualities, train the network, and compute feature importance, together with
`fetch_artifacts.R`, which downloads the trained models.

## Trained models

Five files are archived on Zenodo under DOI 10.5281/zenodo.22880417: four word2vec
models fine-tuned on the evaluations written about instructors of each ethnicity
(`word2vec_white.42B.model`, `word2vec_black.42B.model`,
`word2vec_hispanic.42B.model`, `word2vec_asian.42B.model`) and the trained network
(`model_output_2025-03-27.hdf5`). The models are word2vec trained on the Common
Crawl 42B corpus, which is why the filenames say `word2vec`.

```r
source("fetch_artifacts.R")   # downloads the five files into ./artifacts/
```

## Data

The raw evaluations come from a public dataset (doi: 10.17632/fvtfjyvw7d.2). The
anonymized evaluations used here, with named-quality scores and ethnicity labels
attached and split into training and test sets, are on OSF at https://osf.io/k8ytu/.

## Environment

The network is built with Keras in R on the Python TensorFlow backend, so both
languages have to be installed. The embedding models are trained with gensim.

## License

MIT

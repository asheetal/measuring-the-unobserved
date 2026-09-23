# measuring-the-unobserved

An instrument that measures implicit bias in written evaluations. It reads one
evaluation and returns two things: a score for how strongly the wording points to
the group of the person described, and the named qualities in the writing that
produced that score.

## What is here

The Python and R scripts that fine-tune the word embeddings, score the named
qualities, train the network, and compute feature importance, plus the three
numbered scripts that score new evaluations.

## Trained models

Six files are archived on figshare under DOI 10.6084/m9.figshare.33973036: four word2vec
models fine-tuned on the evaluations written about instructors of each ethnicity
(`word2vec_white.42B.model`, `word2vec_black.42B.model`,
`word2vec_hispanic.42B.model`, `word2vec_asian.42B.model`) and the trained network
(`model_output_2025-03-27.hdf5`). The models are word2vec trained on the Common
Crawl 42B corpus, which is why the filenames say `word2vec`.

`code/2_download_assets.R` downloads the trained network. The four word2vec
models are needed only to retrain the instrument from scratch.

## Data

The raw evaluations come from a public dataset (doi: 10.17632/fvtfjyvw7d.2).
The collation and coding scripts rebuild the analysis dataset from it, and a
fixed seed in the training script reproduces the training and test split.

## Running the package

Run the three scripts in this order.

```bash
cd code
chmod +x 1_environment_check.R 2_download_assets.R 3_run.R   # once, after cloning
./1_environment_check.R
./2_download_assets.R
./3_run.R
```

`1_environment_check.R` reports what is installed, installs the core R packages,
and names what is missing. `2_download_assets.R` downloads the trained network
from figshare. `3_run.R` scores evaluations and names the group the network assigns to each of
the four groups, one row per evaluation.

Four ways to call the scorer:

```bash
./3_run.R                     # two pre-coded evaluations
./3_run.R --complete          # all twenty pre-coded
./3_run.R --full              # two raw evaluations, through the classifiers
./3_run.R --full --complete   # all twenty raw
```

Start with the bare command. It reads `examples_coded.csv`, which already
carries the 125 named qualities, so it loads the trained network and prints
scores in seconds on an ordinary laptop. Nothing else in the package has to work
for this to run.

`--full` reads `examples.csv`, which holds the same twenty evaluations as raw
text, and sends each one through all thirty-two classifiers first. This path
needs a GPU, a Hugging Face account, and the extra downloads described below.
Add `--coder` to `2_download_assets.R` before trying it.

Both files carry `male` and `female` columns. The instrument derives those two
inputs from the instructor's name, and I coded the names ahead of time so that
no instructor name appears in this repository. Anyone scoring their own records
can supply a `professor_name` column instead and let the script derive them.

## Scoring raw text

Scoring evaluations that do not yet carry the named qualities runs thirty-two
classifiers hosted on Hugging Face. They download themselves on first use, but
that first run needs an account, because some of the models are gated.

1. Create a free account at https://huggingface.co/join.
2. Go to https://huggingface.co/settings/tokens and create a token with read
   access.
3. Install the client and log in, which stores the token in `~/.cache/huggingface`:

```bash
pip install --user "huggingface_hub[cli]"
hf auth login
```

4. Grant yourself access to the gated models, described below.

5. Fetch the archived joke detector and pull the rest in advance, so the first
   scoring run does not stall:

```bash
./2_download_assets.R --coder
```

   `Reggie/muppet-roberta-base-joke_detector` has since been withdrawn from
   Hugging Face, so a copy is archived on figshare and this command unpacks it
   into the local cache. The other thirty-one come from Hugging Face:

```bash
for m in \
  CommunicationStyle/Communication_Style \
  Falconsai/fear_mongering_detection \
  Hate-speech-CNERG/english-abusive-MuRIL \
  IDA-SERICS/PropagandaDetection \
  KevSun/Personality_LM \
  KoalaAI/Text-Moderation \
  Reggie/muppet-roberta-base-joke_detector \
  SamLowe/roberta-base-go_emotions \
  Sami92/XLM-R-Large-Sensationalism-Classifier \
  amedvedev/bert-tiny-cognitive-bias \
  bucketresearch/politicalBiasBERT \
  cardiffnlp/twitter-roberta-base-irony \
  cardiffnlp/twitter-roberta-base-sentiment-latest \
  chrlukas/flattery_prediction_text \
  civility-lab/roberta-base-namecalling \
  dejanseo/good-vibes \
  helinivan/english-sarcasm-detector \
  holistic-ai/rejection_detection \
  jackhhao/jailbreak-classifier \
  knkarthick/Action_Decisions \
  leondz/refutation_detector_distilbert \
  madhurjindal/autonlp-Gibberish-Detector-492513457 \
  maximuspowers/bias-type-classifier \
  michellejieli/NSFW_text_classifier \
  padmajabfrl/Gender-Classification \
  parsawar/profanity_model_3.1 \
  pparasurama/raceBERT-ethnicity \
  rafalposwiata/deproberta-large-depression \
  sentinet/suicidality \
  tee-oh-double-dee/social-orientation \
  textdetox/xlmr-large-toxicity-classifier \
  vtiyyal1/empathy_model ; do
  hf download "$m"
done
```

The download is roughly 15 GB and lands in `~/.cache/huggingface/hub`. Once they are
there, score raw text with `./3_run.R ../reviews.csv`, which sets
`HF_HUB_OFFLINE=1` itself so that a model withdrawn later cannot stall the run.

### A GPU is effectively required

Scoring raw text runs thirty-two transformer classifiers over every evaluation.
On a CPU this is slow enough to be impractical for anything beyond a handful of
rows, so treat a GPU as a requirement for this path rather than an optimisation.
Two ways around it: score a CSV that already carries the 125 named-quality
columns, which needs no GPU at all, or run the coding step once on a machine
that has one.

If you are setting up a GPU, follow NVIDIA's own instructions rather than a
distribution package, since the common failure is a driver older than the CUDA
build that torch was compiled against, which crashes R with a segmentation
fault:

1. Install the driver from https://www.nvidia.com/Download/index.aspx and reboot.
2. Install the CUDA toolkit from
   https://developer.nvidia.com/cuda-downloads, choosing the version that
   matches the driver.
3. Check the two agree with `nvidia-smi`, which prints both the driver version
   and the CUDA version it supports.
4. Install the torch build for that CUDA version, following the selector at
   https://pytorch.org/get-started/locally/.
5. Confirm with `python3 -c "import torch; print(torch.cuda.is_available())"`,
   which must print `True`.

`1_environment_check.R` reports whether torch can see a GPU, and `3_run.R` warns
before starting the classifiers without one.

### Gated models

Some authors restrict their models after publication, and a restricted model
stops the run with a 403 and the words "gated repo". Two have gone that way so
far:

- `Falconsai/fear_mongering_detection` is gated. Open
  https://huggingface.co/Falconsai/fear_mongering_detection while logged in,
  press the access button on the model card, and wait for the confirmation,
  which is usually immediate. Then rerun.
- `Reggie/muppet-roberta-base-joke_detector` has been withdrawn altogether, so
  no access request is possible. `./2_download_assets.R --coder` restores it
  from the figshare archive.

The same procedure applies to any other model that becomes gated later: open the
URL printed in the error, request access on that page, and rerun. If the model
has been removed rather than gated, the request button will be absent, and the
run can only continue if someone still holds a cached copy.

Once every model has downloaded once, `HF_HUB_OFFLINE=1` keeps the scorer
working regardless of what happens upstream, and `3_run.R` sets it by default.

## Requirements

R with the `keras`, `reticulate`, and `tidyverse` packages, and a Python that R
can see with TensorFlow installed. `1_environment_check.R` installs the core R
packages from CRAN and prints the exact `pip install` line for anything else. It
creates no virtual environment and changes no existing one.

Two points that cost time otherwise. The trained network was saved with Keras 2,
so on a machine with Keras 3 the scripts set `TF_USE_LEGACY_KERAS` and the check
asks you to install `tf_keras`. Scoring raw text additionally needs `torch` and
`transformers` and runs thirty-two classifiers; on a CPU this takes minutes per
batch, and a torch built for a newer CUDA than the local driver will crash. A
CSV that already carries the 125 named-quality columns avoids all of that, since
it needs only TensorFlow.

## License

MIT

# nlp-models

[![CircleCI](https://circleci.com/gh/epwalsh/nlp-models.svg?style=svg)](https://circleci.com/gh/epwalsh/nlp-models)
[![codecov](https://codecov.io/gh/epwalsh/nlp-models/branch/master/graph/badge.svg)](https://codecov.io/gh/epwalsh/nlp-models)

My NLP research experiments, built on PyTorch within the [AllenNLP](https://github.com/allenai/allennlp) framework.

----

The goal of this project is to provide a model for what a high-quality personal research library could look like. It provides **modularity**, **continuous integration**, **high test coverage**, a code base that emphasizes **readability**, and a host of scripts that make reproducing any experiment here as easy as running a few `make` commands.

## Quick start

The modules implemented here are built and tested nightly against the master branch of AllenNLP. Therefore it is recommended that you install AllenNLP from source. The easiest way to do that is as follows:

```
git clone https://github.com/allenai/allennlp.git && cd allennlp
./scripts/install_requirements.sh
python setup.py develop
```

> NOTE: If you're not already familiar with AllenNLP, I would suggest starting with their [excellent tutorial](https://allennlp.org/tutorials).

After AllenNLP is installed, you can define your own experiments with an AllenNLP model config file, and then run

```bash
make train
# ... follow the prompts to specify the path to your model config and serialization directory.
```

As an example which you should be able to run immediately, I've provided an implementation of **[CopyNet](https://arxiv.org/abs/1603.06393)** and an artificial dataset to experiment with. To train this model, run the following:

```bash
# Extract data.
make data/greetings.tar.gz

# Train model. When prompted for the model file, enter "experiments/greetings/copynet.json".
# This took (~3-5 minutes on a single GTX 1070).
make train
```

> NOTE: All of the model configs in the `experiments/` folder are defined to run on GPU #0. So if you don't have a GPU available or want to use a different GPU, you'll need to modify the `trainer.cuda_device` field in the experiment's config file.

There are several other `make` commands that may come in handy:
- `make test`: Equivalent to running `make typecheck`, `make lint`, `make unit-test`, and `make check-scripts`.
- `make typecheck`: Runs the [mypy](http://mypy-lang.org/) typechecker.
- `make lint`:  Runs [pydocstyle](https://github.com/PyCQA/pydocstyle) and [pylint](https://www.pylint.org/).
- `make unit-test`: Runs all unit tests with [pytest](https://docs.pytest.org/en/latest/).
- `make check-scripts`: Runs a few other scripts that check miscellaneous things not covered by the other tests.
- `make create-branch`: A wrapper around the git functionality to create a new branch and push it upstream. You can name a branch after an issue number with `make create-branch issue=NUM` or give it an arbitrary name with `make create-branch name="my-branch"`.
- `make data/DATASETNAME.tar.gz`: Extract a dataset in the `data/` directory. Just replace `DATASETNAME` with the basename of one of the `.tar.gz` files in that directory.

## Models implemented

**[CopyNet](https://arxiv.org/abs/1603.06393):** A sequence-to-sequence model that incorporates a copying mechanism, which enables the model to copy tokens from the source sentence into the target sentence even if they are not part of the target vocabulary. This architecture has shown promising results on machine translation and semantic parsing tasks. For examples in use, see
- [experiments/greetings/copynet.json](./experiments/greetings/copynet.json)
- [experiments/nl2bash/copynet.json](./experiments/nl2bash/copynet.json)

## Datasets available

For convenience, this project provides a handful of training-ready datasets and scripts to pull and proprocess some other useful datasets. Here is a list so far:

**Greetings:** A simple made-up dataset of greetings (the source sentences) and replies (the target sentences). The greetings are things like "Hi, my name is Jon Snow" and the replies are in the format "Nice to meet you, Jon Snow!". This is completely artificial and is just meant to show the usefullness of the copy mechanism in CopyNet.
```bash
# Extract data.
make data/greetings.tar.gz
```

**[NL2Bash](http://arxiv.org/abs/1802.08979):** A challenging dataset that consists of bash one-liners along with corresponding expert descriptions. The goal is to translate the natural language descriptions into the bash commands.
```bash
# Extract data.
make data/nl2bash.tar.gz
```

**[WMT 2015](http://www.statmt.org/wmt15/translation-task.html):** Hosted with love by [fast.ai](https://www.fast.ai/), this is dataset of 22.5 million English / French sentence pairs that can be used to train an English to French or French to English machine translation system.
```bash
# Download, extract, and preprocess data (big file, may take around 10 minutes).
./scripts/data/pull_wmt.sh
```

# nlp-models

[![CircleCI](https://circleci.com/gh/epwalsh/nlp-models.svg?style=svg)](https://circleci.com/gh/epwalsh/nlp-models)
[![codecov](https://codecov.io/gh/epwalsh/nlp-models/branch/master/graph/badge.svg)](https://codecov.io/gh/epwalsh/nlp-models)

NLP research experiments, built on PyTorch within the [AllenNLP](https://github.com/allenai/allennlp) framework.

----

The goal of this project is to provide an example of a high-quality personal research library. It provides **modularity**, **continuous integration**, **high test coverage**, a code base that emphasizes **readability**, and a host of scripts that make reproducing any experiment here as easy as running a few `make` commands. I also strive to make `nlp-models` useful by implementing practical modules and models that extend AllenNLP. Sometimes I'll contribute pieces of what I work on here back to AllenNLP after it has been thoroughly tested.

## Overview

At a high-level, the structure of this project mimics that of AllenNLP. That is, the submodules in [nlpete](./nlpete) are organized in exactly the same way as in [allennlp](https://github.com/allenai/allennlp/tree/master/allennlp). But I've also provided a set of scripts that automate frequently used command sequences, such as running tests or experiments. The [Makefile](./Makefile) serves as the common interface to these scripts:

- `make train`: Train a model. This is basically a wrapper around `allennlp train`, but provides a default serialization directory and automatically creates subdirectories of the serialization directory for different runs of the same experiment.
- `make test`: Equivalent to running `make typecheck`, `make lint`, `make unit-test`, and `make check-scripts`.
- `make typecheck`: Runs the [mypy](http://mypy-lang.org/) typechecker.
- `make lint`:  Runs [pydocstyle](https://github.com/PyCQA/pydocstyle) and [pylint](https://www.pylint.org/).
- `make unit-test`: Runs all unit tests with [pytest](https://docs.pytest.org/en/latest/).
- `make check-scripts`: Runs a few other scripts that check miscellaneous things not covered by the other tests.
- `make create-branch`: A wrapper around the git functionality to create a new branch and push it upstream. You can name a branch after an issue number with `make create-branch issue=NUM` or give it an arbitrary name with `make create-branch name="my-branch"`.
- `make data/DATASETNAME.tar.gz`: Extract a dataset in the `data/` directory. Just replace `DATASETNAME` with the basename of one of the `.tar.gz` files in that directory.

## Getting started

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

**[WMT 2015](http://www.statmt.org/wmt15/translation-task.html):** Hosted by [fast.ai](https://www.fast.ai/), this is a dataset of 22.5 million English / French sentence pairs that can be used to train an English to French or French to English machine translation system.
```bash
# Download, extract, and preprocess data (big file, may take around 10 minutes).
./scripts/data/pull_wmt.sh
```

## Issues and improvements

If you've found a bug or have any questions, please feel free to [submit an issue on GitHub](https://github.com/epwalsh/nlp-models/issues/new). I always appreciate pull requests as well.

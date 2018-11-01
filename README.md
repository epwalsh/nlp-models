# nlp-models

[![CircleCI](https://circleci.com/gh/epwalsh/nlp-models.svg?style=svg)](https://circleci.com/gh/epwalsh/nlp-models)
[![codecov](https://codecov.io/gh/epwalsh/nlp-models/branch/master/graph/badge.svg)](https://codecov.io/gh/epwalsh/nlp-models)

My NLP research experiments, built on PyTorch within the [AllenNLP](https://github.com/allenai/allennlp) framework.

## Quick start

The models implemented here are built and tested (nightly) against the master branch of AllenNLP. Therefore it is recommended that you install AllenNLP from source. The easiest way to do that is as follows:

```
git clone https://github.com/allenai/allennlp.git && cd allennlp
./scripts/install_requirements.sh
python setup.py develop
```

After AllenNLP is installed, you can define your own experiments with an AllenNLP model config file, and then run

```
make train
```

and follow the prompts to specify the path to your model config and a serialization directory.

## Models implemented

- **[CopyNet](https://arxiv.org/abs/1603.06393):** A sequence-to-sequence model that incorporates a copying mechanism, which enables the model to copy tokens from the source sentence into the target sentence even if they are not part of the target vocabulary. This architecture has shown promising results on machine translation and semantic parsing tasks.

## Datasets available

- **Greetings:** A simple made-up dataset of greetings (the source sentences) and replies (the target sentences). The greetings are things like "Hi, my name is Jon Snow" and the replies are in the format "Nice to meet you, Jon Snow!". This is completely artificial and is just meant to show the usefullness of the copy mechanism in CopyNet.
- **[NL2Bash](http://arxiv.org/abs/1802.08979):** A challenging dataset that consists of bash one-liners along with corresponding expert descriptions. The goal is to translate the natural language descriptions into the bash commands.
- **[WMT 2015](http://www.statmt.org/wmt15/translation-task.html):** Hosted with love by [fast.ai](https://www.fast.ai/), this is dataset of 22.5 million English / French sentence pairs that can be used to train an English to French or French to English machine translation system.

## Experiments

> NOTE: All experiments are defined to run on GPU #0. So if you don't have a GPU available or want to use a different GPU, you'll need to modify the `trainer.cuda_device` field in the experiment's config file.

- **[Greetings dataset with CopyNet](./experiments/greetings/copynet.json)**
```bash
# Extract data.
make data/greetings.tar.gz
# Train model (~3-5 minutes on a single GTX 1070).
make experiments/greetings/copynet.json
```
- **[NL2Bash with CopyNet](./experiments/nl2bash/copynet.json)**
```bash
# Extract data.
make data/nl2bash.tar.gz
# Train model.
make experiments/nl2bash/copynet.json
```
- **[WMT 2015 English to French with CopyNet](./experiments/wmt/en_fr_copynet.json)**
```bash
# Download data from fast.ai (big file, may take around 10 minutes).
./scripts/data/pull_wmt.sh
# Extract data (~1-2 minutes).
make data/wmt.tar.gz
# Combine English and French files into single file (~1-1.5 minutes).
./scripts/data/combine_wmt_en_fr.py
# Train model (oof, this is gonna take a while).
make experiments/wmt/en_fr_copynet.json
```

## TODO

- Implement custom metrics for NL2Bash.

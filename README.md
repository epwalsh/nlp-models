# nlp-models

[![CircleCI](https://circleci.com/gh/epwalsh/nlp-models.svg?style=svg)](https://circleci.com/gh/epwalsh/nlp-models)
[![codecov](https://codecov.io/gh/epwalsh/nlp-models/branch/master/graph/badge.svg)](https://codecov.io/gh/epwalsh/nlp-models)

State-of-the-art and experimental NLP models built on PyTorch within the [AllenNLP](https://github.com/allenai/allennlp) framework.

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

## Datasets

- **Greetings:** A simple made-up dataset of greetings (the source sentences) and replies (the target sentences). The greetings are things like "Hi, my name is Jon Snow" and the replies are in the format "Nice to meet you, Jon Snow!". This is completely artificial and is just meant to show the usefullness of the copy mechanism in CopyNet.
- **[NL2Bash](http://arxiv.org/abs/1802.08979):** A challenging dataset that consists of bash one-liners along with corresponding expert descriptions. The goal is to translate the natural language descriptions into the bash commands.

## Experiments

- **Greetings dataset with CopyNet:** run `make experiments/greetings/copynet.json` to train.
- **[NL2Bash with CopyNet](./experiments/nl2bash/copynet.json):** (WIP) run `make experiments/nl2bash/copynet.json` to train.

## TODO

- Implement custom metrics for NL2Bash.

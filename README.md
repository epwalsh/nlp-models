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
`make train` from the command line.

## Models implemented

- WIP: [CopyNet](https://arxiv.org/abs/1603.06393): A sequence-to-sequence model that incorporates a copying mechanism, which enables the model to copy tokens from the source sentence into the target sentence even if they are not part of the target vocabulary. This architecture has shown promising results on machine translation and semantic parsing tasks.

## Datasets

- [NL2Bash](http://arxiv.org/abs/1802.08979): A challenging dataset that consists of bash one-liners along with corresponding expert descriptions. The goal is to translate the natural language descriptions into the bash commands.

## Experiments

- WIP: [NL2Bash with CopyNet](./experiments/nl2bash/copynet.json): run `make experiments/nl2bash/copynet.json` to train.

## TODO

- Implement beam search for CopyNet

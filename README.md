# nlp-models

[![CircleCI](https://circleci.com/gh/epwalsh/nlp-models.svg?style=svg)](https://circleci.com/gh/epwalsh/nlp-models)
[![Coverage Status](https://coveralls.io/repos/github/epwalsh/nlp-models/badge.svg?branch=master)](https://coveralls.io/github/epwalsh/nlp-models?branch=master)

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

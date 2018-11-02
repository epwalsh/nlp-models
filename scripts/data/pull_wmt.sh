#!/bin/bash

set -e

mkdir -p data/wmt
mkdir -p data/wmt/dev

echo 'Pulling training data'
wget -O data/wmt.tar.gz https://s3.amazonaws.com/fast-ai-nlp/giga-fren.tgz

echo 'Extracting training data'
tar xzfv data/wmt.tar.gz -C data/wmt --strip-components 1

echo 'Combining training data'
./scripts/data/combine_wmt_en_fr.py

echo 'Pulling validation data'
wget -O data/wmt/dev.tar.gz http://www.statmt.org/wmt15/test.tgz

echo 'Extracting validation data'
tar xzfv data/wmt/dev.tar.gz -C data/wmt/dev --strip-components 1

echo 'Converting validation sgm data files to flat text files'
./scripts/data/strip_sgml.pl data/wmt/dev/newsdiscusstest2015-enfr-src.en.sgm | grep -Ev '^$' > data/wmt/dev.en
./scripts/data/strip_sgml.pl data/wmt/dev/newsdiscusstest2015-enfr-ref.fr.sgm | grep -Ev '^$' > data/wmt/dev.fr

echo 'Combining validation data'
./scripts/data/combine_wmt_en_fr.py \
    --en data/wmt/dev.en \
    --fr data/wmt/dev.fr \
    --out data/wmt/english_to_french_dev.tsv

echo 'Cleaning up'
rm -rf ./data/wmt/giga*
rm -rf ./data/wmt/dev*

wc -l ./data/wmt/*.tsv

echo 'Done!'

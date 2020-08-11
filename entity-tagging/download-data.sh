#!/bin/bash

echo ">> Downloading CoNLL-2003 data"
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
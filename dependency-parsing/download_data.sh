#!/bin/bash

if [ ! -f dev.conllu ]
then
  wget -O dev.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
fi
if [ ! -f train.conllu ]
then
  wget -O train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
fi
if [ ! -f test.conllu ]
then
  wget -O test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu
fi
#!/bin/bash

# As we are using the tatoeba dataset: download and preprare the data
if [ ! -f sentences.tar.bz2 ]
then
  echo ">> Downloading sentences.tar.bz2 ..."
  wget http://downloads.tatoeba.org/exports/sentences.tar.bz2
fi
if [ ! -f sentences.csv ]
then
  echo ">> Extracting sentences.tar.bz2"
  tar xjf sentences.tar.bz2
fi

echo ">> Shuffling and converting sentences.csv ..."
awk -F"\t" '{print"__label__"$2" "$3}' < sentences.csv | shuf > sentences.txt

echo ">> Done"

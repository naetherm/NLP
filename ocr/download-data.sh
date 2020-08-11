#!/bin/bash

if [ ! -f image_contest_level_1.tar.gz ]
then
  echo ">> Downloading Baidu OCR dataset"
  wget http://baidudeeplearning.bj.bcebos.com/image_contest_level_1.tar.gz
fi

if [ -f image_contest_level_1.tar.gz ]
then
  echo ">> Extracting the dataset"
  tar -zxf image_contest_level_1.tar.gz
else
  echo "!! The required file for the OCR dataset is not available. Did you download it?"
fi

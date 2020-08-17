# NLP

This repository represents just a collection of machine and deep learning approaches for different tasks in the field of NLP (Natural Language Processing).

## Note

Some codes, e.g. the BERT models for entity tagging and pos tagging are using <= Tensorflow 1.15, I am currently in the process of upgrading those to 2.x.

## Table of Contents

 * [Language Detection](#language-detection)
 * [POS Tagging](#pos-tagging)
 * [Entity Tagging](#entity-tagging)
 * [Sentiment Analysis](#sentiment-analysis)
 * [Word Segmentation](#word-segmentation)
 * [Chatbot](#chatbot)
 
## Content

### [Language Detection](language-detection)

All models were trained and evaluation on the [Tatoeba dataset](http://downloads.tatoeba.org/exports/sentences.tar.bz2).

There are the following implementations:

 * Baseline implementation using the python langdetect module (00_langdetect.py)
 * Character N-Gram implementation (01_nsec_langdetect.py)
 
 
 ### [POS Tagging](pos-tagging)
 
 All models were trained and evaluated on [CONLL POS](https://cogcomp.org/page/resource_view/81) dataset.
 
 There are the following implementations:
 
  * Basic BERT language model approach (10_bert.py)
  
 
 ### [Entity Tagging](entity-tagging)
 
 All models were trained and evaluated on [CONLL POS](https://cogcomp.org/page/resource_view/81) dataset.
 
 There are the following implementations:
 
  * Basic BERT language model approach (10_bert.py)
  
### [Sentiment Analysis](sentiment-analysis)

All models were trained and evaluated on [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/).

There are the following implementations:

  * LSTM model (01_lstm.py)
  * Bidirectional LSTM model (02_bilstm.py)
  
### [Word Segmentation](word-segmentation)

All models were trained on the first 30.000 lines of [Oscar Corpus EN](https://oscar-corpus.com/).

There are the following implementations:

  * LSTM based model (01_lstm.py)
  * Bidirectional LSTM based model (02_bilstm.py)
  * CNN based model (03_cnn.py)
  
### [Chatbot](chatbot)

All models within the chatbot section were trained with the [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
The required files from the corpus were already added to the repository.

There are the following implementations:

  * Basic RNN model (01_seq2seq_rnn.py)
  * LSTM model (02_seq2seq_lstm.py)
  * GRU model (03_seq2seq_gru.py)
  * Bidirectional Basic RNN model (04_seq2seq_birnn.py)
  * Bidirectional LSTM model (05_seq2seq_bilstm.py)
  * Bidirectional GRU model (06_seq2seq_bigru.py)
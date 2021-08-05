# Data

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
import numpy as np
import csv
import collections
import glob
import pandas as pd
tf.__version__
tfds.__version__

tf.keras.backend.clear_session()

if not os.path.exists('./gmb-2.2.0/'):
    print("Please execute the data.sh first")
    exit(1)

# The embedding dimension
EMBEDDING_DIM = 64

# Number of RNN units
HIDDEN_DIM = 100

#batch size
BATCH_SIZE = 90
EPOCHS = 5

SEQ_LEN = 50

fnames = []
for root, dirs, files in os.walk(data_root):
    for filename in files:
        if filename.endswith(".tags"):
            fnames.append(os.path.join(root, filename))
            
fnames[:2]

ner_tags = collections.Counter()
iob_tags = collections.Counter()

def strip_ner_subcat(tag):
    return tag.split("-")[0]


def iob_format(ners):
    iob_tokens = []
    for idx, token in enumerate(ners):
        if token != 'O':  # !other
            if idx == 0:
                token = "B-" + token # start of sentence
            elif ners[idx-1] == token:
                token = "I-" + token  # continues
            else:
                token = "B-" + token
        iob_tokens.append(token)
        iob_tags[token] += 1
    return iob_tokens  

total_sentences = 0
outfiles = []
for idx, file in enumerate(fnames):
    with open(file, 'rb') as content:
        data = content.read().decode('utf-8').strip()
        sentences = data.split("\n\n")
        print(idx, file, len(sentences))
        total_sentences += len(sentences)
        
        with open("./ner/"+str(idx)+"-"+os.path.basename(file), 'w') as outfile:
            outfiles.append("./ner/"+str(idx)+"-"+os.path.basename(file))
            writer = csv.writer(outfile)
            
            for sentence in sentences: 
                toks = sentence.split('\n')
                words, pos, ner = [], [], []
                
                for tok in toks:
                    t = tok.split("\t")
                    words.append(t[0])
                    pos.append(t[1])
                    ner_tags[t[3]] += 1
                    ner.append(strip_ner_subcat(t[3]))
                writer.writerow([" ".join(words), 
                                 " ".join(iob_format(ner)), 
                                 " ".join(pos)])
                                 
# use outfiles param as well
files = glob.glob("./ner/*.tags")

data_pd = pd.concat([pd.read_csv(f, header=None, 
                                 names=["text", "label", "pos"]) 
                for f in files], ignore_index = True)
                
# Keras tokenizer
text_tok = Tokenizer(filters='[\\]^\t\n', lower=False, split=' ', oov_token='<OOV>')

pos_tok = Tokenizer(filters='\t\n', lower=False, split=' ', oov_token='<OOV>')

ner_tok = Tokenizer(filters='\t\n', lower=False,split=' ', oov_token='<OOV>')
                    
text_tok.fit_on_texts(data_pd['text'])
pos_tok.fit_on_texts(data_pd['pos'])
ner_tok.fit_on_texts(data_pd['label'])      

ner_config = ner_tok.get_config()
text_config = text_tok.get_config()

text_vocab = eval(text_config['index_word'])
ner_vocab = eval(ner_config['index_word'])

x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])

# now, pad seqences to a maximum length

x_pad = sequence.pad_sequences(x_tok, padding='post',
                              maxlen=SEQ_LEN)
y_pad = sequence.pad_sequences(y_tok, padding='post',
                              maxlen=SEQ_LEN)
                              
num_classes = len(ner_vocab)+1

Y = tf.keras.utils.to_categorical(y_pad, num_classes=num_classes)
Y.shape

X = x_pad 


# Length of the vocabulary 
vocab_size = len(text_vocab) + 1 

# create training and testing splits
test_size = round(total_sentences / BATCH_SIZE * 0.2)
X_train = X[BATCH_SIZE*test_size:]
Y_train = Y[BATCH_SIZE*test_size:]

X_test = X[0:BATCH_SIZE*test_size]
Y_test = Y[0:BATCH_SIZE*test_size]

# num of NER classes
num_classes = len(ner_vocab)+1


dropout=0.2
model = tf.keras.Sequential([
	Embedding(
		vocab_size, 
		EMBEDDING_DIM, 
		mask_zero=True,
        batch_input_shape=[batch_size, None]),
	Bidirectional(
		LSTM(
			units=HIDDEN_DIM,
            return_sequences=True,
            dropout=dropout,  
            kernel_initializer=tf.keras.initializers.he_normal())),
	TimeDistributed(Dense(HIDDEN_DIM, activation='relu')),
	Dense(num_classes, activation="softmax")
])

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# batch size in eval
model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)



import tensorflow as tf
import pandas as pd 
import re
import os
import io

tf.__version__

INPUT_DIMS=3
UNITS=12

# Data 
lines = io.open('data/SMSSpamCollection').read().strip().split('\n')
lines[0]

spam_dataset = []
count = 0
for line in lines:
  label, text = line.split('\t')
  if label.lower().strip() == 'spam':
    spam_dataset.append((1, text.strip()))
    count += 1
  else:
    spam_dataset.append(((0, text.strip())))

print(spam_dataset[0])
print("Spam: ", count)



df = pd.DataFrame(spam_dataset, columns=['Spam', 'Message'])
# Normalization functions

def message_length(x):
  # returns total number of characters
  return len(x)

def num_capitals(x):
  _, count = re.subn(r'[A-Z]', '', x) # only works in english
  return count

def num_punctuation(x):
  _, count = re.subn(r'\W', '', x)
  return count
  
df['Capitals'] = df['Message'].apply(num_capitals)
df['Punctuation'] = df['Message'].apply(num_punctuation)
df['Length'] = df['Message'].apply(message_length)

train=df.sample(frac=0.8,random_state=1337) #random state is a seed value
test=df.drop(train.index)


# 1-layer neural network model for evaluation
model = tf.keras.Sequential()

# Adds a densely-connected layer with 12 units to the model:
model.add(tf.keras.layers.Dense(
  UNITS, 
  input_dim=INPUT_DIMS, 
  activation='relu'))

# Add a sigmoid layer with a binary output unit:
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = train[['Length', 'Punctuation', 'Capitals']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals']]
y_test = test[['Spam']]

model.fit(x_train, y_train, epochs=10, batch_size=10)
model.evaluate(x_test, y_test)
y_train_pred = model.predict_classes(x_train)
tf.math.confusion_matrix(tf.constant(y_train.Spam), y_train_pred)

y_test_pred = model.predict_classes(x_test)
tf.math.confusion_matrix(tf.constant(y_test.Spam), y_test_pred)

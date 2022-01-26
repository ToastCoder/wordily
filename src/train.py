# WORDILY

# IMPORTING REQUIRED LIBRARIES
import string
import numpy as np
import os 
import tensorflow as tf
from sklearn.model_selection import train_test_split

# DATA IMPORTING
text = open('data/texts.txt', 'r', encoding = 'utf-8')
lines = []
for x in text:
    lines.append(x)

# DATA CLEANING
data = ""
for i in lines:
    data = ' '.join(lines)

data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')

# MAP PUNCTUATION TO SPACE
translate = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translate)

l = []
for i in data.split():
    if i not in l:
        l.append(i)

data = ' '.join(l)

# TOKENIZATION
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 1000, oov_token = '<OOV>')
tokenizer.fit_on_texts([data])
vocab_size = len(tokenizer.word_index) + 1
sequence_data = tokenizer.texts_to_sequences([data])[0]
sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
'''
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

x_train = np.array(X_train)
x_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
'''

# ONE HOT ENCODING
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=vocab_size)

# LSTM BASED NEURAL NETWORK
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=1))
model.add(tf.keras.layers.LSTM(1000, return_sequences=True))
model.add(tf.keras.layers.LSTM(1000))
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))

# PRINTING MODEL SUMMARY
model.summary()

# MODEL TRAINING
#dataset = tf.data.Dataset.from_tensor_slices((X,y))
model.compile(loss="categorical_crossen tropy", optimizer=tf.keras.optimizers.Adam(lr=0.001))
model.fit(X,y,epochs=150, batch_size=64, verbose=1)
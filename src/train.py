# WORDILY

# IMPORTING REQUIRED LIBRARIES
import string
import numpy as np
import os 
import tensorflow as tf

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






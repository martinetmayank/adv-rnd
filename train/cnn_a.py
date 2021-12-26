import tensorflow.keras.backend as K
import multiprocessing

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
import logging

from gensim.models.word2vec import Word2Vec

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer


np.random.seed(1000)

dataset_location = 'dataset/sarcasm_v2.csv'
model_location = 'word2vec/'

corpus = []
labels = []


ip = open(dataset_location, 'r', encoding="utf8")
next(ip)
li = ip.readlines()
random.shuffle(li)

dataset_location = 'dataset/shuffled_sarcasm_v2.csv'

fid = open(dataset_location, "w", encoding="utf8")
fid.writelines(li)
fid.close()

with open(dataset_location, 'r', encoding="utf8") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader)

    for row in csvReader:
        corpus.append(row[3] + " - " + row[4])
        pol = 1 if row[1] == 'sarc' else 0
        labels.append(pol)

print('Corpus size: {}'.format(len(corpus)))

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [
        stemmer.stem(t)
        for t in tkr.tokenize(tweet) if not t.startswith('@')
    ]
    tokenized_corpus.append(tokens)

vector_size = 300
window_size = 10

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

word2vec = Word2Vec(sentences=tokenized_corpus,
                    vector_size=vector_size,
                    window=window_size,
                    negative=20,
                    epochs=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count(),
                    sg=1)


word2vec.save('cnna.model')

X_vecs = word2vec.wv
del corpus

train_size = math.floor(0.8 * len(tokenized_corpus))
test_size = len(tokenized_corpus) - train_size

avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))

print('Average tweet length: {}'.format(
    avg_length / float(len(tokenized_corpus)))
)

print('Max tweet length: {}'.format(max_length))
max_tweet_length = max_length

indexes = set(
    np.random.choice(
        len(tokenized_corpus),
        train_size + test_size, replace=False)
)

X_train = np.zeros(
    (train_size, max_tweet_length, vector_size),
    dtype=K.floatx()
)
Y_train = np.zeros((train_size, 2), dtype=np.int32)
X_test = np.zeros(
    (test_size, max_tweet_length, vector_size),
    dtype=K.floatx()
)
Y_test = np.zeros((test_size, 2), dtype=np.int32)

for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_tweet_length:
            break

        if token not in X_vecs:
            continue

        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]

    if i < train_size:
        Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
    else:
        Y_test[i - train_size, :] = [
            1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]


batch_size = 64
nb_epochs = 10

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
        16,
        kernel_size=2,
        activation='relu',
        padding='same',
        input_shape=(max_tweet_length, vector_size)
    ),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        32,
        kernel_size=2,
        activation='relu',
        padding='same'
    ),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        16,
        kernel_size=2,
        activation='relu',
        padding='same'
    ),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    shuffle=True,
                    epochs=nb_epochs,
                    validation_data=(X_test, Y_test)
                    )

model.save('models/cnn_a.h5')

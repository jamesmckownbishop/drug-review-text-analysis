# -*- coding: utf-8 -*-
"""
This script trains a sentiment analysis model on the Drugs.com drug review dataset 
    provided for public use by Surya Kallumadi and Felix Gräßer.

Each observation is a written review and associated whole number rating between 
    one and ten, inclusive. The model associates word patterns in the reviews with 
    negative (1-4), neutral (5-6), and positive (7-10) ratings. The model is trained 
    using Word2vec neural word embeddings, two stacked recurrent neural network 
    layers and subsequent dense network layers. The model achieves approximately 
    80% accuracy on test data compared to a baseline accuracy of 66%.
"""

import io
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import CuDNNGRU, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from gensim.models import Word2Vec
import html
from nltk import word_tokenize
import numpy as np
import pandas as pd
import random as rn
import requests
import tensorflow as tf
from time import time
from zipfile import ZipFile

def clean_reviews(data_files, file_name):
    '''
    This function loads and cleans a dataset containing text reviews and associated 
    numeric ratings.
    
    Accepts
    -------
    data_files: A ZipFile object containing one or more files
    file_name: A string that is the name of a .csv file with columns named 
        'review' and 'rating'
    
    Returns
    -------
    A pandas DataFrame with columns 'review' and 'rating'
    '''
    df = pd.read_csv(data_files.open(file_name), sep='\t')
    df = df[['review', 'rating']]
    df = df.drop_duplicates()
    df['review'] = df['review'].apply(html.unescape)
    df['review'] = df['review'].str.replace(r'[^a-zA-Z\.\,]', ' ')
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(word_tokenize)
    df = df[df['review'].apply(len) > 0]
    return df

def embed_tokens(tokens, w2v_model):
    '''
    This function maps each string token in a list to a numeric vector
    
    Accepts
    -------
    tokens: A list of strings representing words and other atomic grammatical objects
    w2v_model: A Word2Vec object that maps strings to numeric vectors
    
    Returns
    -------
    An MxN numpy array of numeric values, where M is the number of tokens and N 
    is the vector length defined by the specified model
    '''
    sequence = np.array([get_vector_if_exists(token, w2v_model) for token in tokens])
    return np.array(sequence)

def get_vector_if_exists(token, w2v_model):
    '''
    This function maps a string token to a numeric vector. If the token is not 
    in the specified model's vocabularly, this function returns a vector of zeroes.
    
    Accepts
    -------
    token: A string representing a word or other atomic grammatical object
    w2v_model: A Word2Vec object that maps strings to numeric vectors
    
    Returns
    -------
    A numeric vector with length defined by the specified model
    '''
    try:
        return w2v_model.wv.get_vector(token)
    except KeyError:
        return np.zeros(w2v_model.vector_size)
    
def gen_batch(X, y, w2v_model, bins, max_batch_size=None):
    '''
    This generator yields a batch of 
    
    Accepts
    -------
    X: A pandas Series of lists of string tokens
    y: A numeric pandas Series with the same dimension as X
    w2v_model: A Word2Vec object that maps strings to numeric vectors
    bins: An increasing list of numeric values for categorizing the values in y 
        where all values of y are between the first and last values of bins
    max_batch_size: A positive integer that limits the number of observations in 
        each batch to avoid memory overload or improve performance
    
    Yields
    ------
    A 2-tuple containing a 2-D numpy array of numeric predictor values and a 2-D 
        numpy array of boolean outcome values
    '''
    seq_lens = X.apply(len)
    len_freqs = seq_lens.unique()
    if max_batch_size is None:
        max_batch_size = seq_lens.value_counts().max()
    while(True):
        len_freq_idx = 0
        while(len_freq_idx < len(len_freqs)):
            batch_seq_len = len_freqs[len_freq_idx]
            X_same_len, y_same_len = X[seq_lens == batch_seq_len], y[seq_lens == batch_seq_len]
            idx = 0
            while(idx < X_same_len.shape[0]):
                X_batch, y_batch = X_same_len[idx:idx+max_batch_size], np.array(y_same_len[idx:idx+max_batch_size])
                X_batch = [embed_tokens(tokens, w2v_model) for tokens in X_batch]
                X_batch = np.moveaxis(np.dstack(X_batch), 2, 0)
                y_batch = np.array([np.logical_and(bins[i] < y_batch, y_batch <= bins[i+1]) for i in range(len(bins)-1)]).T
                yield (X_batch, y_batch)
                idx += max_batch_size
            len_freq_idx += 1

start_time = time()

# Initialize constants
TEST_SHARE = 0.25
EPOCHS = 256
PATIENCE = 32
GB_RAM = 6.5
RATING_BINS = [0, 4, 6, 10]

# Initialize hyperparameters
LR = 2**-13
N_DENSE_LAYERS = 2
RECURRENT_LAYER_WIDTH = 64
DENSE_LAYER_WIDTH = 64
EMBED_SIZE = 128
EMBED_WINDOW = 8

# Make results approximately reproducible
SEED = 9999
np.random.seed(SEED)
rn.seed(SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Load training data
data_files = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip')
data_files = ZipFile(io.BytesIO(data_files.content))
df = clean_reviews(data_files, 'drugsComTrain_raw.tsv')

# Compute baseline metrics
bin_shares = pd.cut(df['rating'], RATING_BINS).value_counts(normalize=True)
baseline_loss = -np.sum(bin_shares*np.log(bin_shares))
baseline_acc = bin_shares.max()

# Train embedding
w2v_model = Word2Vec(df['review'], size=EMBED_SIZE, window=EMBED_WINDOW, seed=SEED)

# Separate validation set
val_idxs = np.random.choice(np.arange(df.shape[0]), size=int(df.shape[0]*TEST_SHARE), replace=False)
train_idxs = [i for i in np.arange(df.shape[0]) if i not in val_idxs]
X_val, y_val = df['review'].iloc[val_idxs], df['rating'].iloc[val_idxs]
df = df.iloc[train_idxs]

# Instantiate neural network
input_layer = Input((None, w2v_model.vector_size))
hidden_layer = CuDNNGRU(RECURRENT_LAYER_WIDTH, return_sequences=True)(input_layer)
hidden_layer = CuDNNGRU(RECURRENT_LAYER_WIDTH)(input_layer)
for _ in range(N_DENSE_LAYERS):
    hidden_layer = Dense(DENSE_LAYER_WIDTH, activation='selu')(hidden_layer)
output_layer = Dense(len(RATING_BINS)-1, activation='softmax')(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit neural network
max_batch_size = int(1e9 * GB_RAM / (8 * max(RECURRENT_LAYER_WIDTH, DENSE_LAYER_WIDTH)**2))
steps_per_epoch = int(np.sum(np.ceil(df['review'].apply(len).value_counts() / max_batch_size)))
validation_steps = X_val.apply(len).nunique()
with tf.device('/gpu:0'):
    history = model.fit_generator(gen_batch(df['review'], df['rating'], w2v_model, RATING_BINS, max_batch_size), 
                                  steps_per_epoch=steps_per_epoch, 
                                  epochs=EPOCHS, 
                                  validation_data=gen_batch(X_val, y_val, w2v_model, RATING_BINS), 
                                  validation_steps=validation_steps,
                                  callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
# Load test data
df = clean_reviews(data_files, 'drugsComTest_raw.tsv')

# Measure model performance on test data
steps_per_epoch = int(np.sum(np.ceil(df['review'].apply(len).value_counts() / max_batch_size)))
with tf.device('/gpu:0'):
    metrics = dict(zip(model.metrics_names, model.evaluate_generator(gen_batch(df['review'], df['rating'], w2v_model, RATING_BINS), steps=steps_per_epoch)))
print(f'The model achieves {metrics["acc"]:.1%} accuracy on test data compared to a baseline accuracy of {baseline_acc:.1%}.')

print(f'Run time: {time() - start_time:4.5} seconds')

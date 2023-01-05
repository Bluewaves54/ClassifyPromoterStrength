import keras
from keras import layers
import pandas as pd
import numpy as np


def get_kmers(sequences, kmer=4):
    return_seqs = sequences.copy()
    if kmer <= 1:
        raise ValueError("kmer size must be greater than 1")
    for seq_index, seq in sequences.iteritems():
        kmer_list = []
        for let_index, let in enumerate(seq[:-kmer + 1]):
            kmer_list.append(seq[let_index:let_index + kmer])
        return_seqs[seq_index] = kmer_list
    return return_seqs

def get_2d_kmer(seqs, mnm, mxm):
    return_seqs = []
    for _, val in seqs.iteritems():
        kmer_seqs = []
        for i in range(mnm, mxm+1):
            kmers = list(get_kmers(pd.Series([val]), kmer=i))[0]
            kmers += [kmers[-1] for _ in range(i-1)]
            kmer_seqs.append(kmers)
        return_seqs.append(kmer_seqs)
    
    return pd.Series(return_seqs)

def DCNN(num_features):
    model = keras.Sequential()
    model.add(layers.Dropout(0.1, input_shape=(num_features, 1)))
    model.add(layers.Conv1D(32, 3, activation='softsign', input_shape=(num_features, 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='softsign'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

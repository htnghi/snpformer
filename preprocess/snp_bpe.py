from snp_tokenizing import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace

#Tokenizing using overlapping kmer
seq_overlap = seqs2kmer_overlapping(join_data, 3)

# ---------------------------------------
# Function for using BPE to embed
# ---------------------------------------
def prepare_tokenizer_trainer(voc_size=100):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size = voc_size)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(iterator, vocab_size):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(vocab_size)
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer    

def batch_iterator(dataset):
    batch_size = 500
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def train_biological_tokenizer(seq, vocab_size, results_path, max_length):
    # Create (training) new tokenizer from the dataset
    print(batch_iterator(seq))
    tokenizer = train_tokenizer(batch_iterator(seq), vocab_size)
    
    tokenizer.enable_padding(length=max_length) #padding to max_len
    

    # Saving trained tokenizer to the file path
    tokenizer.save(os.path.join(results_path, "tokenizer.json")) # dictionary of {"G": 1, "A": 2, ....}
    # print(tokenizer.get_vocab_size()) # get vocab_size

    # Tokenizing data
    # by assign idices to each token
    def encode(X):
        result = []
        for x in X: #loop each sample in data(i.e. X_train ['SK GE EL FT G.', '...',...])
            ids = tokenizer.encode(x).ids #assign idices to each token[13, 29, 5, 52, 18]  + padding
            if len(ids) > max_length:
                ids = ids[:max_length] # trunct sequences if len(sample)>max_len
            result.append(ids)
        return result
    seq = encode(seq)
    return seq

vocab_size = 10000
results_path = './'
max_length = 2100

X = train_biological_tokenizer(join_data, vocab_size, results_path, max_length)
print(X[0])

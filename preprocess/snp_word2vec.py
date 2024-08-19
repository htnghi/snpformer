from transformer_SNP.preprocess.example_tokenizing import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from math import sin, cos, sqrt, log

#Tokenizing using overlapping kmer
seq_overlap = seqs2kmer_overlapping(join_data, 3)

# ---------------------------------------
# Function for using Word2vec to embed
# ---------------------------------------
def index_Word2vec(sequences):
    
    ## Train Word2Vec model
    word2vec_model = Word2Vec(sentences=sequences, vector_size=100, window=5, min_count=1)
    # word2vec_model.build_vocab(gen_word)
    word2vec_model.train(sequences, total_examples=len(seq_overlap), epochs=1)
    
    # Get embedding for each amino acid
    # embeddings = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key} # write in shor way
    vocab = word2vec_model.wv.index_to_key  # assuming index_to_key provides vocabulary words
                                            # vocab: ['CAC', 'GGT', 'AGT', ..., 'TTT', 'AAC']
    embeddings = dict.fromkeys(vocab, None)  # create dictionary with all words as keys and None as default values
                                             # {'CAC': None, 'GGT': None, ..., 'TTT': None, 'AAC': None}
    # Fill the dictionary with actual embeddings
    for word in vocab:
        embeddings[word] = word2vec_model.wv[word]

    # Initialize an empty list to store the word embeddings
    sequence_embeddings = []
    # Loop through each word in the sentence
    for sequence in sequences:
        current_seq = []
        for word in sequence:
            current_seq.append(embeddings[word])
        sequence_embeddings.append(current_seq)

    return embeddings, sequence_embeddings

embeddings, sequence_embeddings = index_Word2vec(seq_overlap)

# i.e. dictionary of 64, each array is 100: {'CGA': [1.2 , -7.2, ..], 'GAC': [..], 'ACA': [..], ..}
print("Token to Word2vec Mapping:", embeddings)

# Output the embedding for a specific token (e.g., 'AGT')
print("Embedding for k-mer 'AGT':", embeddings['AGT'].shape)
print(sequence_embeddings)


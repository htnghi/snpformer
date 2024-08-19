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
# Function for using indexed to embed
# ---------------------------------------
def assign_token_index(sequences):
    token_to_index = {}  # Initialize an empty dictionary to store token-to-index mappings
    index = 0  # Initialize an index counter
    indexed_kmer_sequences = []  # Initialize an empty list to store indexed k-mer sequences
    # Iterate over each k-mer sequence
    for k_mers in sequences:
        indexed_k_mers = []  # Initialize an empty list to store indexed k-mers for current sequence
        # Assign index to each token
        for k_mer in k_mers:
            if k_mer not in token_to_index:
                token_to_index[k_mer] = index
                index += 1
            indexed_k_mers.append(token_to_index[k_mer])
        indexed_kmer_sequences.append(indexed_k_mers)
    return indexed_kmer_sequences, token_to_index

indexed_kmer_sequences, token_to_index = assign_token_index(seq_overlap)

# i.e. # print(mapped_sequences) # [[23, 34, 21, ...], [...], ...]
print("K-mer Sequences:", indexed_kmer_sequences)

# i.e. dictionary of 64: {'CGA': 0, 'GAC': 1, 'ACA': 2, 'CAG': 3,...}
print("Token to Index Mapping:", token_to_index)




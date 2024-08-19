import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
from gensim.models import Word2Vec
from math import sin, cos, sqrt, log

# ---------------------------------------
# Load file
# ---------------------------------------
X_train_csv, y_train_csv = pd.read_csv('../data/x_train_onehot.csv'), pd.read_csv('../data/y_train_onehot.csv')

X_train, y_train = X_train_csv.iloc[:,1:].to_numpy(), y_train_csv.iloc[:,1].to_numpy()

# Create a list of sequences 
# from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
join_data = [''.join(seq) for seq in X_train]

# ---------------------------------------
# 1. Tokenizing using overlapping k-mer
# ---------------------------------------
# Overlapping k-mer
def seqs2kmer_overlapping(seqs, kmer):

    seq_kmers = [] # Initialize an empty list to store the k-mers for each sequence
    # Iterate over each sequence in the input list
    for s in seqs:
        # Generate k-mers for the current sequence and append them to seq_kmers

        # ouput: [['ACG', 'CGT', 'GTA', 'TAC', ..., 'ACG', 'CGT'], ['CGT, ..., 'CTA'], ...]
        # seq_kmers.append([s[i:i+kmer] for i in range(0, len(s)-kmer+1)])
        # output: [[ACG CGT GTA TAC ... ACG CGT], [CGT ... CTA]]
        seq_kmers.append(' '.join([s[i:i+kmer] for i in range(len(s) - kmer + 1)]))
    # Return the list of k-mers for each sequence
    return seq_kmers

seq_overlap = seqs2kmer_overlapping(join_data, 3)
# print('Sequence after overlapping', seq_overlap[0])

# ---------------------------------------
# 2. Tokenizing using non-overlapping k-mer
# ---------------------------------------
# Non-overlapping k-mer
def seqs2kmer_nonoverlapping(seqs, kmer):

    seq_kmers = [] 
    for s in seqs:
        seq_kmers.append([s[i:i+kmer] for i in range(0, len(s), kmer)])

    return seq_kmers

seq_overlap = seqs2kmer_nonoverlapping(join_data, 3)
# print('Sequence after non-overlapping', seq_overlap[1])

# ---------------------------------------
# 3. Tokenizing into each nucleotides (A, T, C, G)
# ---------------------------------------
# Split sequences into lists of nuclotides
seq_nuc = [list(sequence) for sequence in join_data]
# print('Sequence after splitting into nucleotides', seq_nuc[1])






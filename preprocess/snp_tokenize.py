import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
from gensim.models import Word2Vec


# ---------------------------------------
# 1. Tokenizing using overlapping k-mer
def seqs2kmer_overlap(seqs, kmer):

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

# ---------------------------------------
# 2. Tokenizing using non-overlapping k-mer
def seqs2kmer_nonoverlap(seqs, kmer):

    seq_kmers = [] 
    for s in seqs:
        # ouput: [['ACG', 'CGT', 'GTA', 'TAC', ..., 'ACG', 'CGT'], ['CGT, ..., 'CTA'], ...]
        # seq_kmers.append([s[i:i+kmer] for i in range(0, len(s), kmer)])
        # output: [[ACG CGT GTA TAC ... ACG CGT], [CGT ... CTA]]
        seq_kmers.append(' '.join([s[i:i+kmer] for i in range(0, len(s), kmer)]))
    return seq_kmers

def seqs2kmer_nonoverlap_w2v(seqs, kmer):

    seq_kmers = [] 
    for s in seqs:
        # ouput: [['ACG', 'CGT', 'GTA', 'TAC', ..., 'ACG', 'CGT'], ['CGT, ..., 'CTA'], ...]
        seq_kmers.append([s[i:i+kmer] for i in range(0, len(s), kmer)])
        # output: [[ACG CGT GTA TAC ... ACG CGT], [CGT ... CTA]]
        # seq_kmers.append(' '.join([s[i:i+kmer] for i in range(0, len(s), kmer)]))
    return seq_kmers

# ---------------------------------------
# 3. Tokenizing into each nucleotides (A, T, C, G)
def seqs2nuc(seqs):
    return [list(sequence) for sequence in seqs]

# ---------------------------------------
# Function to tokenize sequence
# ---------------------------------------
def snp_tokenizer(tokenize_type, seqs, kmer):

    seq_tokenizing = []

    if tokenize_type == 'overlap':
        seq_tokenizing = seqs2kmer_overlap(seqs, kmer)
    
    elif tokenize_type == 'non_overlap':
        seq_tokenizing = seqs2kmer_nonoverlap(seqs, kmer)

    elif tokenize_type == 'nuc':
        seq_tokenizing = seqs2nuc(seqs)

    return seq_tokenizing




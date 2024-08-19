import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from math import sin, cos, sqrt, log
import itertools
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace

# ---------------------------------------
# 1. K-mer
# ---------------------------------------
def prepare_tokenizer_trainer_kmer():
    tokenizer = Tokenizer(WordLevel())
    trainer = WordLevelTrainer(special_tokens = ['[PAD]'])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer_kmer(iterator):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer_kmer()
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer    

def batch_iterator_kmer(dataset):
    batch_size = 500
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def kmer_embed(seqs, kmer):
    
    tokenizer_path = os.path.join('./', "tokenizer_kmer" + str(kmer) +".json")
    if not os.path.exists(tokenizer_path):
        print('Tokenizer {} - mer is not exited, create K-mer tokenize'.format(kmer))
        # Create (training) new tokenizer from the dataset
        # print(batch_iterator(seqs))
        tokenizer = train_tokenizer_kmer(batch_iterator_kmer(seqs))
        # tokenizer.enable_padding(length=max_length) # padding to max_len

        # Saving trained tokenizer to the file path
        tokenizer.save(tokenizer_path) # dictionary of {"G": 1, "A": 2, ....}
        # print(tokenizer.get_vocab_size()) # get vocab_size
    else:
        print('Tokenizer {} - mer is already created, just load it'.format(kmer))
        tokenizer = Tokenizer.from_file(tokenizer_path) # If the tokenizer already exist, we load it
    
    return tokenizer # Returns the loaded tokenizer or the trained tokenizer


# Tokenizing data
# by assign idices to each token
def encode(X, tokenizer, max_length):
    result = []
    tokenizer.enable_padding(length=max_length)
    # loop each sample in data(i.e. X_train ['SK GE EL FT G.', '...',...])
    for x in X:

        # assign idices to each token[13, 29, 5, 52, 18] + padding
        ids = tokenizer.encode(x).ids 

        if len(ids) > max_length:
            ids = ids[:max_length] # trunct sequences if len(sample)>max_len
        # else:
        #     tokenizer.enable_padding(length=max_length)

        result.append(ids)

    return result
# ---------------------------------------
# 1. Function for using indexed to embed
# ---------------------------------------
def token_embed(seqs):
    token_to_index = {}  # Initialize an empty dictionary to store token-to-index mappings
    index = 1  # Initialize an index counter
    indexed_kmer_sequences = []  # Initialize an empty list to store indexed k-mer sequences
    # Iterate over each k-mer sequence
    for k_mers in seqs:
        indexed_k_mers = []  # Initialize an empty list to store indexed k-mers for current sequence
        # Assign index to each token
        for k_mer in k_mers:
            if k_mer not in token_to_index:
                token_to_index[k_mer] = index
                index += 1
            indexed_k_mers.append(token_to_index[k_mer])
        indexed_kmer_sequences.append(indexed_k_mers)
    return indexed_kmer_sequences

# indexed_kmer_sequences, token_to_index = token_embed(seqs)

# i.e. # print(mapped_sequences) # [[23, 34, 21, ...], [...], ...]
# print("K-mer Sequences:", indexed_kmer_sequences)

# i.e. dictionary of 64: {'CGA': 0, 'GAC': 1, 'ACA': 2, 'CAG': 3,...}
# print("Token to Index Mapping:", token_to_index)


# ---------------------------------------
# 2. Function for using Word2vec to embed
# ---------------------------------------
def word2vec_embed_not_missing_key(seqs, word2vec_model=None, k=6):
    # Tokenize each sequence into characters (or other tokens like k-mers)
    # tokenized_seqs = [list(seq) for seq in seqs]  # Tokenizing by characters
    # tokenized_seqs = [ [seq[i:i+k] for i in range(len(seq) - k + 1)] for seq in seqs]
    tokenized_seqs = [ [seq[i:i+k] for i in range(0, len(seq), k)] for seq in seqs]


    # Train Word2Vec model
    # If a Word2Vec model is not provided, train a new one
    if word2vec_model is None:
        word2vec_model = Word2Vec(sentences=tokenized_seqs, vector_size=100, window=10, min_count=1)
        word2vec_model.train(tokenized_seqs, total_examples=len(tokenized_seqs), epochs=100)  # Train the model
    
    # Convert the sequences to embeddings
    sequence_embeddings = []
    for sequence in tokenized_seqs:
        current_seq = [word2vec_model.wv[word] for word in sequence]
        sequence_embeddings.append(current_seq)
    
    # Convert the list of lists to a NumPy array
    sequence_embeddings_array = np.array(sequence_embeddings)
    
    return sequence_embeddings_array, word2vec_model

def word2vec_embed(seqs, word2vec_model=None, k=6, vector_size=100):
    # Tokenize each sequence into k-mers (6-mers in this case)
    tokenized_seqs = [[seq[i:i+k] for i in range(0, len(seq), k)] for seq in seqs]
    print(len(tokenized_seqs))


    # If a Word2Vec model is not provided, train a new one
    if word2vec_model is None:
        word2vec_model = Word2Vec(sentences=tokenized_seqs, vector_size=vector_size, window=10, min_count=1)
        word2vec_model.train(tokenized_seqs, total_examples=len(tokenized_seqs), epochs=100)
    
    # Initialize a zero vector for missing k-mers
    zero_vector = np.zeros(vector_size)
    print(len(tokenized_seqs))

    # Convert the sequences to embeddings using the provided or trained model
    sequence_embeddings = []
    for sequence in tokenized_seqs:
        # print(sequence)
        current_seq = []
        for word in sequence:
            print(word)
            
            if word in word2vec_model.wv:
                current_seq.append(word2vec_model.wv[word])
                print(current_seq)
                exit()
            else:
                current_seq.append(zero_vector)  # Use zero vector for unseen k-mers
        sequence_embeddings.append(current_seq)
    
    # Convert the list of lists to a NumPy array
    sequence_embeddings_array = np.array(sequence_embeddings)
    
    return sequence_embeddings_array, word2vec_model

def word2vec_embed_list(list_seqs, word2vec_model=None, k=6, vector_size=120):
    sequence_embed_list = []
    word2vec_model_list = []
    
    for i, seqs in enumerate(list_seqs):

        # Tokenize each sequence into k-mers (6-mers in this case)
        # output: [[ACG CGT GTA TAC ... ACG CGT], [CGT ... CTA]]
        # tokenized_seqs = [[seq[i:i+k] for i in range(0, len(seq), k)] for seq in seqs]
        # transformed_seqs = [[seq[i:i+k] for i in range(0, len(seq), k)] for seq in seqs]
        transformed_seqs = [row.tolist() for row in seqs]

        # If a Word2Vec model is not provided, train a new one
        if word2vec_model is None:
            current_model = Word2Vec(sentences=transformed_seqs, vector_size=vector_size, window=10, min_count=1)
            current_model.train(transformed_seqs, total_examples=len(transformed_seqs), epochs=100)
            word2vec_model_list.append(current_model)
        else:
            # Use the corresponding pre-trained Word2Vec model
            current_model = word2vec_model[i]
        
        # Initialize a zero vector for missing k-mers
        zero_vector = np.zeros(vector_size)
        
        # Convert the sequences to embeddings using the provided or trained model
        sequence_embeddings = []
        for sequence in transformed_seqs:
            current_seq = []
            for word in sequence:
                if word in current_model.wv:
                    current_seq.append(current_model.wv[word])
                else:
                    current_seq.append(zero_vector)  # Use zero vector for unseen k-mers
            sequence_embeddings.append(current_seq)
            
        # Convert the list of lists to a NumPy array
        sequence_embeddings_array = np.array(sequence_embeddings)
    
        sequence_embed_list.append(sequence_embeddings_array)
    
    # Only return the model list if we trained new models
    if word2vec_model is None:
        return sequence_embed_list, word2vec_model_list
    else:
        return sequence_embed_list, None
    
# ---------------------------------------
# 3. Function for using BPE to embed
# input seq using seq (not in k_mer)
# ---------------------------------------
def prepare_tokenizer_trainer(vocab_size):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens = ['[PAD]'], vocab_size=vocab_size)
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

def BPE_embed(seqs, train_ratio, chr_index, vocab_size: int=2048):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))
    
    tokenizer_path = os.path.join('./preprocess/tokenizer/BPE/ratio_' + label_train + '_' + label_test, "tokenizer_BPE" + str(chr_index) + "_" + str(vocab_size) + ".json")
    if not os.path.exists(tokenizer_path):
        print('Tokenizer BPE {} - vocab size:{} for ratio {} is not exited, create BPE tokenize'.format(chr_index, vocab_size, train_ratio))
        # Create (training) new tokenizer from the dataset
        # print(batch_iterator(seqs))
        tokenizer = train_tokenizer(batch_iterator(seqs), vocab_size)
        # tokenizer.enable_padding(length=max_length) # padding to max_len

        # Saving trained tokenizer to the file path
        tokenizer.save(tokenizer_path) # dictionary of {"G": 1, "A": 2, ....}
        # print(tokenizer.get_vocab_size()) # get vocab_size
    else:
        print('Tokenizer BPE {} - vocab size:{} for ratio {} is already created, just load it'.format(chr_index, vocab_size, train_ratio))
        tokenizer = Tokenizer.from_file(tokenizer_path) # If the tokenizer already exist, we load it
    
    return tokenizer # Returns the loaded tokenizer or the trained tokenizer

# Check and choose the max_len of sequence in config (i.e. if reasult =180 --> max_len = 200) 
def choose_max_length(X, tokenizer):
    max_len_src = 0

    for x in X:
        src_ids = tokenizer.encode(x).ids
        # print('len src_ids', len(src_ids))
        max_len_src = max(max_len_src, len(src_ids))
    # return print(f'Max length of sample in SNP dataset: {max_len_src}')
    return max_len_src

# Tokenizing data
# by assign idices to each token
def encode_BPE(X, tokenizer, max_length):
    result = []
    tokenizer.enable_padding(length=max_length)
    # loop each sample in data(i.e. X_train ['SK GE EL FT G.', '...',...])
    for x in X:

        # assign idices to each token[13, 29, 5, 52, 18] + padding
        ids = tokenizer.encode(x).ids 

        if len(ids) > max_length:
            ids = ids[:max_length] # trunct sequences if len(sample)>max_len
        # else:
        #     tokenizer.enable_padding(length=max_length)

        result.append(ids)

    return result
# seqs = encode(seqs)
# return seqs

# X = BPE_embed(join_data, vocab_size, results_path, max_length)
# print(X[0])


# ---------------------------------------
# Function to tokenize as k-mer
# ---------------------------------------
def generate_kmers(k):
    """Generate all possible k-mers of length k using the nucleotides A, T, C, and G."""
    nucleotides = ['A', 'T', 'C', 'G']
    kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    return kmers

def create_vocab_index(min_k, max_k):
    """Create a dictionary mapping k-mers of lengths from min_k to max_k to unique indices."""
    vocab_index = {"[PAD]": 0}
    current_index = 1  # Start indices from 1 since 0 is reserved for "PAD"
    for k in range(min_k, max_k + 1):
        kmers = generate_kmers(k)
        for kmer in kmers:
            vocab_index[kmer] = current_index
            current_index += 1
    return vocab_index

def load_kmer_tokenizer(kmer):
    tokenizer_path = os.path.join('./', f"tokenizer_kmer_{kmer}.json")
    if not os.path.exists(tokenizer_path):
        print(f'Tokenizer {kmer}-mer does not exist. Creating k-mer tokenizer...')
        # Normally, you would train a new tokenizer here
        # For this example, let's assume we're reusing the existing vocab_index
        vocab_index = create_vocab_index(1, kmer)
        with open(tokenizer_path, 'w') as json_file:
            json.dump(vocab_index, json_file, indent=4)
        print(f'Tokenizer {kmer}-mer created and saved.')
    else:
        print(f'Tokenizer {kmer}-mer already exists. Loading it...')
        with open(tokenizer_path, 'r') as json_file:
            vocab_index = json.load(json_file)
    
    return vocab_index

# Step 4: Function to encode DNA sequences using the k-mer vocabulary
def encode_sequences(kmer_sequences, tokenizer, max_length):
    result = []
    pad_id = 0  # Index of "PAD" token is 0

    for kmer_seq in kmer_sequences:
        ids = []
        for kmer in kmer_seq.split():
            if kmer in tokenizer:
                ids.append(tokenizer[kmer])
            else:
                ids.append(pad_id)  # Use pad_id if k-mer is not in tokenizer

        # Apply padding or truncation
        if len(ids) < max_length:
            ids.extend([pad_id] * (max_length - len(ids)))  # Pad with pad_id
        else:
            ids = ids[:max_length]  # Truncate to max_length
        
        result.append(ids)

    return result

def find_max_kmer_length(kmer_sequences, tokenizer):
    """Find the maximum length of k-mer sequences."""
    max_kmer_length = 0

    for kmer_seq in kmer_sequences:
        ids = []
        for kmer in kmer_seq.split():
            ids.append(tokenizer[kmer])
        max_len_src = max(max_kmer_length, len(ids))
    # return print(f'Max length of sample in SNP dataset: {max_len_src}')
    return max_len_src

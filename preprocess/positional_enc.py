from math import sin, cos, sqrt, log
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=100, dropout=0.1):
        """
        Positional Embedding or Positional Encoding
        The general idea here is to add positional encoding to the input embedding
        before feeding the input vectors to the first encoder/decoder
        The positional embedding must have the same embedding dimension as in the embedding vectors
        For the positional encoding we use sin and cos

        :param embed_dim: the size of the embedding, this must be the same as in embedding vector
        :param max_seq_len: the maximum sequence length (max sequence of words)
        :param dropout: the dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim # Dimensionality of model (also named d_model in some codes)
        self.max_seq_len = max_seq_len # Maximum sequence length
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

        # Creating a positional encoding matrix of shape (max_seq_len, d_model) filled with zeros
        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)

        # Creating a tensor representing positions (0 to seq_len - 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1) # # Transforming 'position' into a 2D tensor['seq_len, 1']
        
        # Creating the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in pe
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to even indices in pe
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = positional_encoding.unsqueeze(0)

        # we use register_buffer to save the "pe" parameter to the state_dict
        # Buffer is a tensor not considered as a model parameter
        self.register_buffer('pe', pe)

    def pe_sin(self, position, i):
        return sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:, : x.size(1)].shape)

        # Adding positional encoding to the input tensor X
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x) # Dropout for regularization
from math import sin, cos, sqrt, log
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
        Embedding class to convert a word into embedding space (numerical representation)
        :param vocab_size: the vocabulary size
        :param embed_dim: the embedding dimension

        example: if we have 1000 vocabulary size and our embedding is 512,
        then the embedding layer will be 1000x512

        suppose we have a batch size of 64 and sequence of 15 words,
        then the output will be 64x15x512 ((samples, seq_length, embed_dim))
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim # Dimension of model
        self.vocab_size = vocab_size # Size of the vocabulary
        self.embed = nn.Embedding(vocab_size, embed_dim) # Pytorch layer that converts integer indices to dense embeddings

    def forward(self, x):
        """
        forward pass
        :param x: the word or sequence of words
        :return: the numerical representation of the input
        """

        # splitted_x = torch.tensor_split(x, [218, 778, 1323, 1871], dim=1)
        # x0 = splitted_x[0]
        # x1 = splitted_x[1]
        # x2 = splitted_x[2]
        # x3 = splitted_x[3]
        # x4 = splitted_x[4]

        # Normalizing the variance of the embeddings
        # output0 = self.embed(x0) * sqrt(self.embed_dim)
        # output1 = self.embed(x1) * sqrt(self.embed_dim)
        # output2 = self.embed(x2) * sqrt(self.embed_dim)
        # output3 = self.embed(x3) * sqrt(self.embed_dim)
        # output4 = self.embed(x4) * sqrt(self.embed_dim)

        # print(f"Embedding shape: {output.shape}") #shape (samples, seq_length, embed_dim)
        # output = torch.cat((output0, output1, output2, output3, output4), 1)
        # print('Output embedding', output.shape)

        output = self.embed(x) * sqrt(self.embed_dim)
        # print('Output embedding', output.shape)
        return output
    

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
        # x = torch.cat((x0, x1, x2, x3, x4), 1)
        # print('Output Positional encoding', x.shape)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # print('Output Positional encoding', x.shape)
        # Dropout for regularization
        return self.dropout(x) 
    

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len, dropout):

        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim # Dimension of model
        self.max_seq_len = max_seq_len 
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch, seq_len, embed_dim = x.size()

        positions = torch.arange(seq_len).to(device=x.device)
        positions = self.pos_embedding(positions)[None, :, :].expand(batch, seq_len, embed_dim)
        x_add = x + positions

        return self.dropout(x_add)

# Functions used for Rotary Position Embedding
class RoPE(nn.Module):
    def __init__(self, emb_dim: int, theta: Optional[float] = 10000.0):
        super(RoPE, self).__init__()
        self.theta = theta
        self.emb_dim = emb_dim

    def compute_freq_cis(self, device, seq_len: int):
        # Frequency calculation. Shape of t_theta: (emb_dim // 2,)
        t_theta = 1.0 / (self.theta ** (torch.arange(0, self.emb_dim, 2, device=device)[:self.emb_dim // 2] / self.emb_dim))
        # Create position indices. Shape of t:
        t = torch.arange(seq_len, device=device)
        # Frequency Matrix Calculation: compute the product of t_theta and t. Shape: (seq_len, emb_dim // 2)
        freqs = torch.outer(t, t_theta)
        # torch.polar constructs complex numbers from these magnitudes and angles
        # form of each elements: e^i_theta = cos(theta) + i*sin(theta)
        # Shape: (seq_len, emb_dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, device, query: torch.Tensor, key: torch.Tensor):
        # print('query shape', query.shape)
        # print('Query', query[0])
        b, h, t, c = query.size()
        query = query.to(device)
        key = key.to(device)
        freq_cis = self.compute_freq_cis(device, t) # ensure freq_cis matches the sequence length

        # Reshape and convert to complex
        query_complex = torch.view_as_complex(query.float().reshape(b, h, t, c // 2, 2))
        key_complex = torch.view_as_complex(key.float().reshape(b, h, t, c // 2, 2))

         # Apply RoPE (Element-wise Multiplication with freq_cis): query_complex[i]∗freq_cis[i]
    # Convert Back to Real and Flatten
        q_rot = torch.view_as_real(query_complex * freq_cis).flatten(3)
        k_rot = torch.view_as_real(key_complex * freq_cis).flatten(3)
        return q_rot.type_as(query), k_rot.type_as(key)



class RotaryEmbedding(nn.Module):
    def __init__(self, device, key_size: int, rescaling_factor: Optional[float] = None):
        super(RotaryEmbedding, self).__init__()
        UPPER_FREQ = 10000.0

        if rescaling_factor is None:
            self._inv_freq = 1.0 / (UPPER_FREQ ** (torch.arange(0, key_size, 2, device=device).float() / key_size))
        else:
            updated_base = UPPER_FREQ * (rescaling_factor ** (key_size / (key_size - 2)))
            self._inv_freq = 1.0 / (updated_base ** (torch.arange(0, key_size, 2, device=device).float() / key_size))

    def _compute_cos_sin_tables(self, device, heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print('Qeury shape', heads.shape) #(batch, h, seq_len, d_k) #([32, 8, 1667, 56])
        # print('Query', heads[0])
        seq_len = heads.size(2)
        # print('Seq_len', seq_len)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # print('t', t.shape) #([1667])
        freqs = torch.einsum("i,j->ij", t, self._inv_freq)
        # print('freqs', freqs.shape) #([1667, 28])
        emb = torch.cat((freqs, freqs), dim=-1).to(heads.dtype)
        # print('emb', emb.shape) #([1667, 56])

        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        # print('cos_cached', cos_cached.shape) #([1, 1, 1667, 56])
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        # print('sin_cached', sin_cached.shape) #([1, 1, 1667, 56])

        return cos_cached, sin_cached

    def _apply_rotary_pos_emb(self, heads: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_size = heads.size(-1)
        x1 = heads[..., : head_size // 2]
        # print('x1', x1.shape) #([32, 8, 1667, 28])
        x2 = heads[..., head_size // 2 :]
        heads_rotated = torch.cat((-x2, x1), dim=-1)
        # print('heads_rotated', heads_rotated.shape) #([32, 8, 1667, 56])

        embedded_heads = (heads * cos) + (heads_rotated * sin)
        return embedded_heads

    def forward(self, device, query_heads: torch.Tensor, key_heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_heads = query_heads.to(device)
        key_heads = key_heads.to(device)
        cos, sin = self._compute_cos_sin_tables(device, query_heads)
        return (
            self._apply_rotary_pos_emb(query_heads, cos, sin),
            self._apply_rotary_pos_emb(key_heads, cos, sin),
        )


############################################################################################
class RotaryEmbedding_Minh(nn.Module):

    def __init__(self, device, dim: int, seq_len: int, dim_embedding_pct: float=1.0, base: int=10000):
        """
        Rotary Positional Embedding
        The implementation scheme is reffered to https://github.com/dingo-actual/infini-transformer, which is based on the paper
        "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al. (https://arxiv.org/abs/2104.09864).

        :param dim: the dimention of key/value of the attention layer
        :param max_seq_len: the maximum length/total length of the full sequence
        :param dim_embedding_pct: the percentage of the total embedding dimension is used for the positional embeddings; must be within the interval (0, 1]. Defaults to 0.5
        :param base: base is used to calculate thetas, detault is set 10000.
        """
        super(RotaryEmbedding_Minh, self).__init__()

        self.dim = dim
        self.effective_dim = int(dim * dim_embedding_pct)
        # print("[RotaryEmbedding] dim={}, effective_dim={}".format(self.dim, self.effective_dim))
        self.seq_len = seq_len
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        self.last_offset = 0

        # initialize the theta matrix
        # self._calculate_thetas()

        # initialize sin component indices for input tensor
        # indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg

    def _calculate_thetas(self, device, input_seq_len, offset: int=0) -> None:

        # Calculate matrix of angles: thetas[i,j] = base^(-2 * ceil(i/2)) * (j + offset)
        thetas = torch.repeat_interleave(
            (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1, device=device))).unsqueeze(-1).repeat((1, input_seq_len)), 
                repeats=2, 
                dim=0
            )
            
        # Multiply by index positions, then transpose to get correct shape
        thetas *= torch.arange(1 + offset, input_seq_len + 1 + offset, device=device).unsqueeze(0)
        self.thetas = thetas.transpose(0, 1).unsqueeze(0).unsqueeze(0)

    def _compute_cos_sin_tables(self, device, heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = heads.size(2)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # print('t', t.shape) #([1667])
        freqs = torch.einsum("i,j->ij", t, self._inv_freq)
        # print('freqs', freqs.shape) #([1667, 28])
        emb = torch.cat((freqs, freqs), dim=-1).to(heads.dtype)
        # print('emb', emb.shape) #([1667, 56])

        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        # print('cos_cached', cos_cached.shape) #([1, 1, 1667, 56])
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        # print('sin_cached', sin_cached.shape) #([1, 1, 1667, 56])

        return cos_cached, sin_cached

    def _apply_rotary_pos_emb(self, heads: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_size = heads.size(-1)
        x1 = heads[..., : head_size // 2]
        # print('x1', x1.shape) #([32, 8, 1667, 28])
        x2 = heads[..., head_size // 2 :]
        heads_rotated = torch.cat((-x2, x1), dim=-1)
        # print('heads_rotated', heads_rotated.shape) #([32, 8, 1667, 56])

        embedded_heads = (heads * cos) + (heads_rotated * sin)
        return embedded_heads

    # def forward(self, query_heads: torch.Tensor, key_heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     cos, sin = self._compute_cos_sin_tables(query_heads)
    #     return (
    #         self._apply_rotary_pos_emb(query_heads, cos, sin),
    #         self._apply_rotary_pos_emb(key_heads, cos, sin),
    #     )

    def forward(self, device, x, offset, select_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        x = x.to(device)
        _,_, input_seq_len,_ = x.size()

        if offset != self.last_offset:
            self._calculate_thetas(device, input_seq_len, offset)
            self.last_offset = offset
            cos_sin_recalculated = True
        else:
            cos_sin_recalculated = False
        
        # print("[forward] passed cos_sin_recalculated={}".format(cos_sin_recalculated))
        
        if self.dim_embedding_pct < 1.0:
            x_pos = x[..., :self.effective_dim]
            x_pass = x[..., self.effective_dim:]
        else:
            x_pos = x
        # print("[forward] passed x_pos={} ".format(x_pos.shape))
        
        if not cos_sin_recalculated:
            self._calculate_thetas(device, input_seq_len, offset)
            self.last_offset = offset
        
        # print("[forward] passed not cos_sin_recalculated: last_offset={}".format(self.last_offset))
        # print("[forward] self.thetas shape={}".format(self.thetas.shape))

        x_cos = self.thetas.cos().repeat(1, x_pos.size(1), 1, 1) * x_pos
        # print("[forward] x_cos={}".format(x_cos.shape))

        x_sin = x_pos[..., self.ixs_sin]
        # print("[forward] x_sin={}".format(x_sin.shape))
        x_sin[..., self.ixs_sin_neg] = -x_sin[..., self.ixs_sin_neg]
        # print("[forward] x_sin_neg={}".format(x_sin.shape))
        x_sin *= self.thetas.sin().repeat(1, x_pos.size(1), 1, 1)
        # print("[forward] passed xcos={}, xsin={}".format(x_cos.shape, x_sin.shape))

        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        # print("[forward] passed out={}".format(out.shape))

        return out
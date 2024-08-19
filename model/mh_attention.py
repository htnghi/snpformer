from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Literal, Optional, Union

from model.embed_layer import *

class MultiHeadAttention(nn.Module):

    def __init__(self, device, d_model: int, heads: int, dropout: float, use_rope: Optional[bool] = True):
        """
        Multi-Head Attention class
        :param embed_dim: the embedding dimension
        :param heads: the number of heads, default equals 8
        """
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.use_rope = use_rope
        self.d_model = d_model # 512 by default
        self.heads = heads #8 by default
        
        # We ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % heads == 0, 'd_model is not divisible by h'
        
        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = d_model // heads # d_k formula, # 512 / 8 = 64 by default
        
        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False) # # the Query weight metrix
        self.w_k = nn.Linear(d_model, d_model, bias=False) # W_k
        self.w_v = nn.Linear(d_model, d_model, bias=False) # W_v
        # Define the weight matrice W0
        # fully connected layer: 8*64x512 or 512x512
        self.w_o = nn.Linear(d_model, d_model) # W_o
        
        self.dropout = nn.Dropout(dropout) # Dropout layer to avoid overfitting

        if self.use_rope:
            # self.rope_model = RoPE(self.d_k, theta=10000.0)
            self.rope_model = RotaryEmbedding(self.device, self.d_k)
    
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask):# mask => When we want certain words to NOT interact with others, we "hide" them
        
        d_k = query.shape[-1] # The last dimension of query, key, and value
        
        # We calculate the Attention(Q,K,V) as in the formula in the image above 
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # @ = Matrix multiplication sign in PyTorch
        # print('Attention score at beggining', attention_scores) #batch_size, heads, q_len, k_len

        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None: # If a mask IS defined...
            # print('Attention score shape', attention_scores.shape)
            attention_scores.masked_fill_(mask == 0, -1e9) # Replace each value where mask is equal to 0 by -1e9
        attention_scores = attention_scores.softmax(dim = -1) # Applying softmax
        if dropout is not None: # If a dropout IS defined...
            attention_scores = dropout(attention_scores) # We apply dropout to prevent overfitting
            
        return (attention_scores @ value), attention_scores # Multiply the output matrix by the V matrix, as in the formula

    def forward(self, x, mask):

        # Input of size: batch_size x sequence length x embedding dims
        # batch_size, seq_len, d_model = x.shape
        # Projection into query, key, value: (batch, seq_len, d_model)
        query = self.w_q(x) # Q' matrix
        key = self.w_k(x) # K' matrix
        value = self.w_v(x) # V' matrix
        
        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension

        # Apply RoPE if enabled
        if self.use_rope:
            # query, keys = RoPE(self.device, self.freq_cis, q_segment, k_segment)
            # query, key = self.rope_model(self.device, q_segment, k_segment)
            query, key = self.rope_model(self.device, query, key)
            
        # Obtaining the output and the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.dropout, mask)
        
        # Obtaining the H matrix
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        output = self.w_o(x)  # H x W0 = (32x10x8x64) x (32x8x64x512) = (32x10x512)
        
        return output
    
class MultiHeadAttentionChunk(nn.Module):

    def __init__(self, device, d_model: int, heads: int, dropout: float, use_rope: Optional[bool] = True):
    
        super(MultiHeadAttentionChunk, self).__init__()
        self.device = device
        self.use_rope = use_rope

        self.d_model = d_model
        self.heads = heads
        
        assert d_model % heads == 0, 'd_model is not divisible by h'
        
        self.d_k = d_model // heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.use_rope:
            # self.rope_model = RoPE(self.d_k, theta=10000.0)
            self.rope_model = RotaryEmbedding(self.device, self.d_k)
    
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask):
        
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores

    def forward(self, x, mask=None):

        query = self.w_q(x)
        key   = self.w_k(x)
        value = self.w_v(x)
        
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2)

        # Apply RoPE if enabled
        if self.use_rope:
            # query, keys = RoPE(self.device, self.freq_cis, q_segment, k_segment)
            # query, key = self.rope_model(self.device, q_segment, k_segment)
            query, key = self.rope_model(self.device, query, key)

        x, self.attention_scores = MultiHeadAttentionChunk.attention(query, key, value, self.dropout, mask=None)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        output = self.w_o(x)
        
        return output
    
class InfiniteAttention(nn.Module):
    def __init__(self, 
                 device,
                 d_model: int, 
                 heads: int, 
                 seq_len: int,
                 segment_len: int,
                 dropout: float,
                 update_rule: Optional[str] = 'linear',
                 use_rope: Optional[bool] = True 
                 ):
        """
        Implementation the Infinite Attention. Link: https://arxiv.org/pdf/2404.07143
        :param d_model: the embedding dimension
        :param heads: the number of heads
        :param segment_len: Segment length
        :param dropout: dropout rate
        :param update_rule: Type of memory update rule to use ("linear" or "delta")
        """
        super(InfiniteAttention, self).__init__()
        
        # Input params
        self.device = device
        self.d_model = d_model
        self.heads = heads
        self.seq_len = seq_len
        self.segment_len = segment_len
        self.dropout = dropout
        self.update_rule = update_rule
        self.use_rope = use_rope
        
        # Ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % heads == 0, 'dimensions of the model is divisible by the number of heads'
        
        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = d_model // heads # d_k formula, # 512 / 8 = 64 by default

        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False, device=device) 
        self.w_k = nn.Linear(d_model, d_model, bias=False, device=device) 
        self.w_v = nn.Linear(d_model, d_model, bias=False, device=device) 
        # Define the weight matrice W0
        self.w_o = nn.Linear(d_model, d_model, device=device) # W_o

        # Initialize parameter betas for weighted average of dot-product and memory-based attention:
        self.betas = nn.Parameter(torch.randn((1, heads, 1, self.d_k), device=device))
        
        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # Initialize RoPE if used
        if self.use_rope:
            # self.freq_cis = compute_freq_cis(device, self.d_k, segment_len, theta=10000.0)
            # self.rope_model = RoPE(self.d_k, theta=10000.0)
            self.rope_model = RotaryEmbedding(self.device, self.d_k)
            # self.rope_model = RotaryEmbedding_Minh(device, self.d_k, seq_len)


    @staticmethod
    def scaleddot_product_attention(query, key, value, dropout: nn.Dropout, mask):# mask => When we want certain words to NOT interact with others, we "hide" them
        
        d_k = query.shape[-1] # The last dimension of query, key, and value
        
        # We calculate the Attention(Q,K,V) as in the formula in the image above 
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # @ = Matrix multiplication sign in PyTorch
        # print('Attention score at beggining', attention_scores) #batch_size, heads, q_len, k_len

        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None: # If a mask IS defined...
            # print('Attention score shape', attention_scores.shape)
            attention_scores.masked_fill_(mask == 0, -1e9) # Replace each value where mask is equal to 0 by -1e9
        attention_scores = attention_scores.softmax(dim = -1) # Applying softmax
        if dropout is not None: # If a dropout IS defined...
            attention_scores = dropout(attention_scores) # We apply dropout to prevent overfitting
            
        return (attention_scores @ value), attention_scores # Multiply the output matrix by the V matrix, as in the formula

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor): mask to hide padding tokens (batch_size, 1, ,1, seq_len )

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """

        # Input of size: batch_size x sequence length x embedding dims
        batch_size, seq_len, _ = x.shape

        # Initialize parameter for memomory and normalization z:
        memory = torch.zeros(1, self.heads, self.d_k, self.d_k).to(self.device)
        # z = torch.ones(batch_size, self.heads, self.d_k, 1) / self.dim_key
        z = torch.zeros(batch_size, self.heads, self.d_k, 1).to(self.device)

        # 1. Projection
        query = self.w_q(x) # Q' matrix
        key = self.w_k(x) # K' matrix
        value = self.w_v(x) # V' matrix

        # 2. Split into smaller matrices for different attention heads
        # Splitting d_model (third dim) into h parts
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.heads, self.d_k).transpose(1,2)
        key = key.view(batch_size, seq_len, self.heads, self.d_k).transpose(1,2) 
        value = value.view(batch_size, seq_len, self.heads, self.d_k).transpose(1,2) 

        # 3. Split into number of segments
        n_segments, rem = divmod(seq_len, self.segment_len)
        n_segments += 1 if rem > 0 else 0

        output = []
        # Loop through every segments
        for idx in range(n_segments):
            start_position = idx * self.segment_len
            end_position = min(start_position + self.segment_len, self.seq_len)

            # Extract segment: 
            # Q, K, V of each segment(batch, h, segment_len, d_k)
            q_segment = query[:, :, start_position:end_position, :]
            k_segment = key[:, :, start_position:end_position, :]
            v_segment = value[:, :, start_position:end_position, :]

            # Get padding_mask in correlation with each segment
            mask_segment = mask[:, :, :, start_position:end_position]
            
            # Apply RoPE if enabled
            if self.use_rope:
                # q_pos, k_pos = RoPE(self.device, self.freq_cis, q_segment, k_segment)
                # q_pos, k_pos = self.rope_model(self.device, q_segment, k_segment)
                # q_pos, k_post = self.rope_model(q_segment, k_segment)
                q_pos, k_pos = self.rope_model(self.device, q_segment, k_segment)
                # q_pos = self.rope_model(self.device, q_segment, offset=start_position)
                # k_pos = self.rope_model(self.device, k_segment, offset=start_position)
            
            # 4. Scaled-dot Product Attention: (batch, h, segment_len, d_k)
            if self.use_rope:
                A_dot, self.attention_scores = InfiniteAttention.scaleddot_product_attention(q_pos, k_pos, v_segment, self.dropout, mask_segment)
            else:
                A_dot, self.attention_scores = InfiniteAttention.scaleddot_product_attention(q_segment, k_segment, v_segment, self.dropout, mask_segment)

            # 5. Compressive Memory
            # 5.1. Memory retrieval
            act_q = (nn.functional.elu(q_segment) + 1.0)
            # using paper-Equation(3): (batch, h, segment_len, d_k)
            # add 1e-9 to avoid division by zero
            A_mem = (act_q @ memory) / ((act_q @ z) + 1e-9)

            # 5.2. Memory update
            act_k = (nn.functional.elu(k_segment) + 1.0)
            act_k_T = act_k.transpose(-2, -1)

            # Equation_5
            if self.update_rule == 'delta':
                memory = memory + act_k_T @ (v_segment - (act_k @ memory) / ((act_k @ z) + 1e-9))
            elif self.update_rule == 'linear':
                memory = memory + act_k_T @ v_segment
            
            # Update Normalization (Equation 4)
            z = z + act_k.sum(dim=-2, keepdim=True).transpose(-2, -1)
            
            # 6. Long-term context injection (weighted average of dot-product and memory-based attention)
            # Equation 6: (batch, h, segment_len, d_k)
            attention = (F.sigmoid(self.betas) * A_mem) + ((1 - F.sigmoid(self.betas)) * A_dot)

            # 7. Obtaining the H matrix
            # (batch, h, segment_len, d_k) --> (batch, segment_len, h, d_k) --> (batch, segment_len, d_model)
            attention = attention.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        
            # 8. Projextion output: (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            attention_output = self.w_o(attention)

            # 9. Append ouput
            output.append(attention_output)
    
        # Concatenated full sequence
        out = torch.concat(output, dim=1)
        
        return out
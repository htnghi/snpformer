import torch
import torch.nn as nn
from model.utils import replicate
from model.mh_attention import *
from model.embed_layer import Embedding, PositionalEncoding


class Encoder(nn.Module):

    def __init__(self,
                 device,
                 embed_dim,
                 heads,
                 expansion_factor,
                 dropout,
                 use_rope: Optional[bool] = True
                 ):
        """
        The Transformer Block used in the encoder and decoder as well

        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(device, embed_dim, heads, dropout, use_rope= use_rope)  # the multi-head attention
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.init_layer_norm()

        self.dropout = nn.Dropout(dropout)

        # The fully connected feed-forward layer, 
        # apply two linear transformations with a ReLU activation in between
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # e.g: 512x(4*512) -> (512, 2048)
            nn.GELU(),  # ReLU activation function
            nn.Linear(embed_dim * expansion_factor, embed_dim),  # e.g: 4*512)x512 -> (2048, 512)
        )

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def init_layer_norm(self):
        for layer_norm in [self.layer_norm1, self.layer_norm2]:
            bound = 1 / (layer_norm.weight.size()[0] ** 1 / 2)
            torch.nn.init.uniform_(layer_norm.weight, -bound, bound)

    def forward(self, x, mask):

        """
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # PostNorm: (attention + skip-connection) -> norm1 -> (mlp + skip-connection) -> norm2 -> dropout
        # att_out = self.attention(x, mask)
        # att_norm = self.layer_norm1(att_out+x)

        # mlp_out = self.mlp(att_norm)
        # mlp_norm = self.layer_norm2(mlp_out + att_norm)

        # out = self.dropout(mlp_norm)

        # PreNorm: norm1 -> (attention + skip-connection) -> dropout -> norm2 -> (mlp + skip-connection) -> dropout
        att_out = x + self.dropout(self.attention(self.layer_norm1(x), mask))
        out = att_out + self.dropout(self.mlp(self.layer_norm2(att_out)))

        return out
    
class EncoderChunk(nn.Module):

    def __init__(self,
                 device,
                 embed_dim,
                 heads,
                 expansion_factor,
                 dropout,
                 use_rope: Optional[bool] = True
                 ):
        """
        The Transformer Block used in the encoder and decoder as well

        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(EncoderChunk, self).__init__()

        self.attention = MultiHeadAttentionChunk(device, embed_dim, heads, dropout, use_rope= use_rope)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.init_layer_norm()

        # the fully connected feed-forward layer, apply two linear 
        # transformations with a ReLU activation in between
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def init_layer_norm(self):
        for layer_norm in [self.layer_norm1, self.layer_norm2]:
            bound = 1 / (layer_norm.weight.size()[0] ** 1 / 2)
            torch.nn.init.uniform_(layer_norm.weight, -bound, bound)

    def forward(self, x, mask=None):

        # PostNorm: (attention + skip-connection) -> norm1 -> (mlp + skip-connection) -> norm2 -> dropout
        # att_out = self.attention(x, mask)
        
        # att_norm = self.dropout(self.layer_norm1(att_out+x))
        
        # mlp_out = self.mlp(att_norm)
        # mlp_norm = self.layer_norm2(mlp_out + att_norm)

        # out = self.dropout(mlp_norm)

        # PreNorm: norm1 -> (attention + skip-connection) -> dropout -> norm2 -> (mlp + skip-connection) -> dropout
        att_out = x + self.dropout(self.attention(self.layer_norm1(x), mask))
        out = att_out + self.dropout(self.mlp(self.layer_norm2(att_out)))

        # Serialize-Norm: 
        # att_out = x + self.attention(self.layer_norm1(x), mask)
        # out = x + self.mlp(self.layer_norm2(att_out))
        # out = self.dropout(out)
        
        return out

class EncoderBlock(nn.Module):

    def __init__(self,
                 seq_len,
                 vocab_size,
                 embed_dim,
                 num_blocks,
                 expansion_factor,
                 heads,
                 dropout
                 ):
        """
        The Encoder part of the Transformer architecture
        it is a set of stacked encoders on top of each others, in the paper they used stack of 6 encoders

        :param seq_len: the length of the sequence, in other words, the length of the words
        :param vocab_size: the total size of the vocabulary
        :param embed_dim: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each encoder
        :param heads: the number of heads in each encoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super(EncoderBlock, self).__init__()
        # define the embedding: (vocabulary size x embedding dimension)
        self.embedding = Embedding(vocab_size, embed_dim)
        # define the positional encoding: (embedding dimension x sequence length)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # define the set of blocks
        # so we will have 'num_blocks' stacked on top of each other
        self.blocks = replicate(Encoder(embed_dim, heads, expansion_factor, dropout), num_blocks)

    def forward(self, x):
        out = self.positional_encoder(self.embedding(x))
        for block in self.blocks:
            out = block(out, out, out)
        
        # print(out.shape) # output shape: batch_size x seq_len x embed_size, e.g.: 32x12x512
        return out
    
class InfiniteTransformer(nn.Module):
    """Transformer layer with compressive memory.
    https://github.com/dingo-actual/infini-transformer/blob/main/infini_transformer/transformer.py
    """

    def __init__(
        self,
        dim_input: int,
        expansion_factor: int,
        dim_key: int,
        dim_value: int,
        num_heads: int,
        # activation: str,
        segment_len: int,
        update: str = "linear",
        causal: bool = False,
        # position_embedder: Optional[PositionEmbeddings] = None,
        position_embedder: Optional[PositionalEncoding] = None,
        init_state_learnable: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module for the CompressiveMemory. Defaults to None.
            init_state_learnable (bool, optional): Whether the initial state of the CompressiveMemory should be learnable. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(InfiniteTransformer, self).__init__()
        
        # If sampling_factor passed to kwargs, use it, otherwise set to None
        sampling_factor = kwargs.get("sampling_factor", None)
        
        # Multi-head attention
        self.attn = CompressiveMemory(
            dim_input=dim_input, 
            dim_key=dim_key, 
            dim_value=dim_value, 
            num_heads=num_heads, 
            segment_len=segment_len, 
            sampling_factor=sampling_factor,
            update=update, 
            causal=causal, 
            position_embedder=position_embedder, 
            init_state_learnable=init_state_learnable)
        # MLP
        # if activation not in ACTIVATIONS:
        #     raise ValueError(f"Invalid activation function: {activation}")
        # if activation in ["swiglu", "geglu", "ffnglu", "ffngeglu", "ffnswiglu"]:
        #     act = ACTIVATIONS[activation](dim_hidden)
        # else:
        #     act = ACTIVATIONS[activation]()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, expansion_factor * dim_input),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(expansion_factor * dim_input, dim_input),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        # Apply multi-head attention, followed by MLP and layer normalization with residual connection.
        x_ = self.attn(x)
        x_ = self.mlp(x_)
        x_ = self.layer_norm(x_ + x)

        return x_

class InfiniteEncoder(nn.Module):
    def __init__(self,
                 device,
                 d_model: int,
                 heads: int,
                 expansion_factor: int,
                 seq_len: int,
                 segment_len: int,
                 update_rule: Optional[str] = 'linear',
                 use_rope: Optional[bool] = True,
                 dropout: float = 0.0
                 ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module for the CompressiveMemory. Defaults to None.
            init_state_learnable (bool, optional): Whether the initial state of the CompressiveMemory should be learnable. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(InfiniteEncoder, self).__init__()

        
        # Multi-head attention
        self.attention = InfiniteAttention(
            device=device,
            d_model=d_model, 
            heads=heads, 
            seq_len = seq_len,
            segment_len=segment_len,
            dropout=dropout,
            update_rule=update_rule,
            use_rope= use_rope
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),
            # nn.Dropout(dropout),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(expansion_factor * d_model, d_model)
            # nn.Dropout(dropout)
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        # self.init_layer_norm()

        self.dropout = nn.Dropout(dropout)
    
    # def init_layer_norm(self):
    #     for layer_norm in [self.layer_norm1, self.layer_norm2]:
    #         bound = 1 / (layer_norm.weight.size()[0] ** 1 / 2)
    #         torch.nn.init.uniform_(layer_norm.weight, -bound, bound)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # (attention + skip-connection) -> norm1 -> (mlp + skip-connection) -> norm2 -> dropout

        att_out = self.attention(x, mask)
        att_norm = self.layer_norm1(att_out+x)

        mlp_out = self.mlp(att_norm)
        mlp_norm = self.layer_norm2(mlp_out + att_norm)

        out = self.dropout(mlp_norm)

        # norm1 -> (attention + skip-connection) -> dropout -> norm2 -> (mlp + skip-connection) -> dropout

        # att_out = x + self.dropout(self.attention(self.layer_norm1(x), mask))
        # out = att_out + self.dropout(self.mlp(self.layer_norm2(x)))

        return out
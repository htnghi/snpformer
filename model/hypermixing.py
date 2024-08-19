import math
from typing import Optional, Tuple
from typing import Callable, Optional, Tuple

import torch
from torch import nn

import math
from typing import Callable, Optional

import torch
from torch import nn

class HyperMixing(nn.Module):
    """This class implements multi-head HyperMixing.

    >>> inputs = torch.rand([8, 60, 512])
    >>> net = HyperMixing(512, 2048, num_heads=8)
    >>> outputs = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied: bool = False,
        num_heads: int = 1,
        max_length: int = 3000,
        token_information: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_layer_norm: bool = True,
        hypernetwork_bias: bool = False,
        
    ) -> None:
        """Initialize multi-head HyperMixing.

        Arguments
        ----------
        input_output_dim : int (channel dimensions)
            number of features in keys, queries, and values
        hypernet_size : int
            determines the size of the hidden layer of the token-mixing MLP.
        tied : bool
            If True, then the generated weight matrices of the token-mixing MLP are tied.
        num_heads : int
            parallel token-mixing MLPs.
        max_length : int
            Maximum number of input tokens. Needed for generating sufficiently large position embeddings.
        token_information : Callable
            Function that encodes token information. Defaults to positional encoding.
        use_layer_norm : bool
            If True, layer normalization is used.
        """
        super().__init__()
        self.input_output_dim = input_output_dim
        self.hypernetwork_bias = hypernetwork_bias
        self.hyper = _HyperNetwork(
            input_output_dim,
            hypernet_size,
            tied=tied,
            num_heads=num_heads,
            keep_output_size=False,
            hypernetwork_bias=hypernetwork_bias,
        )
        self.activation = nn.GELU()
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_output_dim)

        # add pos encoding or a custom token_information function
        if token_information is None:
            self.token_information = _PositionalEncoding(input_output_dim, max_length)
        else:
            self.token_information = token_information

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through HyperMixer.

        Arguments
        ----------
        query : torch.Tensor
            (B, M, E) where M is the number of queries, B is the batch size,
            E is the embedding dimension.
        key : torch.Tensor
            (B, N, E) where N is the number of keys/values, B is the batch size,
            E is the embedding dimension.
        value : torch.Tensor
            (B, N, E) where N is the number of keys/values, B is the batch size,
            E is the embedding dimension.
        query_padding_mask : torch.Tensor
            (B, M) where B is the batch size, M is the number of queries.
            If a ByteTensor is provided, the non-zero positions (padding token) will be ignored
            while the position with the zero positions will be unchanged. If a
            BoolTensor is provided, the positions with the value of True (padding token) will be
            ignored while the position with the value of False will be unchanged.
        key_padding_mask : torch.Tensor
            (B, N) where B is the batch size, N is the number of keys/values.
            If a ByteTensor is provided, the non-zero positions will be ignored
            while the position with the zero positions will be unchanged. If a
            BoolTensor is provided, the positions with the value of True will be
            ignored while the position with the value of False will be unchanged.
        Returns
        -------
        torch.Tensor
            output tensor
        """
        bsize = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        if value.size(1) != key.size(1):
            raise ValueError("Number of values not equal to number of keys!")

        if key_padding_mask is not None:
            key_float_mask = torch.logical_not(key_padding_mask).unsqueeze(-1).float()
            key = key * key_float_mask
            value = value * key_float_mask
        if query_padding_mask is not None:
            query_float_mask = (
                torch.logical_not(query_padding_mask).unsqueeze(-1).float()
            )
            query = query * query_float_mask

        # add token information (like position) before passing to hypernetwork
        hyp_in = self.token_information(key)
        hyp_out = self.token_information(query)

        # [bsize, num_heads, key_len/query_len, hypernet_size // num_heads]
        W1, W2 = self.hyper(hyp_in, hyp_out)

        # mask the weights
        if key_padding_mask is not None:
            W1 = W1 * key_float_mask.unsqueeze(1)
        if query_padding_mask is not None:
            W2 = W2 * query_float_mask.unsqueeze(1)

        # reshape the num_heads into the batch dimension for parallelizing
        value = value.transpose(1, 2)  # [bsize, input_output_dim, key_len]
        value = value.reshape(
            (bsize * self.num_heads, self.input_output_dim // self.num_heads, key_len)
        )  # [bsize * num_heads, input_output_dim // num_heads, key_len]
        W1 = W1.reshape((bsize * self.num_heads, key_len, -1))
        W2 = W2.reshape((bsize * self.num_heads, query_len, -1))

        # we stick the token-mixing MLP together manually
        out = _mlp_pass_from_components(value, W1, W2, self.activation)

        # concatenate heads
        out = out.reshape((bsize, self.input_output_dim, query_len))

        # transpose back
        out = out.transpose(1, 2)

        # apply layer norm on outputs of the TM-MLP
        if self.use_layer_norm:
            out = self.layer_norm(out)

        return out


class _HyperNetwork(nn.Module):
    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied: bool = False,
        num_heads: int = 1,
        keep_output_size: bool = True,
        hypernetwork_bias: bool = False,
    ) -> None:
        super().__init__()

        self.tied = tied
        self.w1_gen = _ParallelMLPs(
            input_output_dim,
            input_output_dim,
            output_size=hypernet_size,
            num_mlps=num_heads,
            keep_output_size=keep_output_size,
            bias=hypernetwork_bias,
        )
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = _ParallelMLPs(
                input_output_dim,
                input_output_dim,
                output_size=hypernet_size,
                num_mlps=num_heads,
                keep_output_size=keep_output_size,
                bias=hypernetwork_bias,
            )

    def forward(self, w1_input: torch.Tensor, w2_input: torch.Tensor):
        """Forward pass through the HyperNetwork.

        input_tensor : [batchsize, max_positions, d]
        The HyperNetwork is supposed to generate an MLP of the form W_2(GELU(W1^T x)), where
        W1 : N -> k and W2 : M -> k, so it has to return W1 and W2
        """
        W1 = self.w1_gen(w1_input)
        if self.tied:
            if not torch.equal(w1_input, w2_input):
                raise ValueError("Tied weights but queries != keys")

            W2 = W1
        else:
            W2 = self.w2_gen(w2_input)

        return W1, W2


class _ParallelMLPs(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_mlps: int = 1,
        keep_output_size: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.original_in_size = input_size
        self.original_out_size = output_size

        if input_size % num_mlps != 0:
            raise ValueError("Input size must be divisible by num_mlps")
        if output_size % num_mlps != 0:
            raise ValueError("Output size must be divisible by num_mlps")
        if hidden_size % num_mlps != 0:
            raise ValueError("Hidden size must be divisible by num_mlps")

        input_size = input_size // num_mlps

        if not keep_output_size:
            output_size = output_size // num_mlps
        hidden_size = hidden_size // num_mlps

        self.input_size = input_size
        self.output_size = output_size

        self.num_mlps = num_mlps

        self.fc1_weights = nn.Parameter(torch.empty(num_mlps, hidden_size, input_size))
        self.fc2_weights = nn.Parameter(torch.empty(num_mlps, output_size, hidden_size))
        nn.init.xavier_uniform_(self.fc1_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_weights, gain=math.sqrt(2.0))

        self.bias = bias
        if self.bias:
            self.fc1_biases = nn.Parameter(torch.empty(num_mlps, hidden_size))
            self.fc2_biases = nn.Parameter(torch.empty(num_mlps, output_size))
            nn.init.xavier_uniform_(self.fc2_biases, gain=math.sqrt(2.0))
            nn.init.xavier_uniform_(self.fc1_biases, gain=math.sqrt(2.0))

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        # x [bsize, seq_len, num_features]

        bsize = x.size(0)
        seq_len = x.size(1)

        x = x.reshape((bsize, seq_len, self.num_mlps, self.input_size))
        x = torch.einsum("blmf,mhf->bmlh", x, self.fc1_weights)
        if self.bias:
            x = x + self.fc1_biases.unsqueeze(0).unsqueeze(2)
        x = self.activation(x)
        x = torch.einsum("bmlh,mfh->bmlf", x, self.fc2_weights)
        if self.bias:
            x = x + self.fc2_biases.unsqueeze(0).unsqueeze(2)

        return x


def _mlp_pass_from_components(
    out, W1: torch.Tensor, W2: torch.Tensor, activation: nn.Module
) -> torch.Tensor:
    # we stick the token MLP together manually

    '''
    Input
        out: (batch_size, sequence_length, output_dim)
        W1: (batch_size, input_dim, hidden_dim)
        W2: (batch_size, hidden_dim, output_dim)
    return
        output: (batch_size, sequence_length, output_dim)
    '''
    # torch.bmm is matrix multiplication
    out = torch.bmm(out, W1) #(batch_size, sequence_length, hidden_dim)
    out = activation(out)
    out = torch.bmm(out, W2.transpose(1, 2))

    return out


class _PositionalEncoding(nn.Module):
    """Adds sinoidal position embeddings to the input."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * (i + 1)) / d_model))
                    )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"x shape: {x.shape}")
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        # pe = self.pe[:, 1 : seq_len + 1]
        pe = self.pe[:, :seq_len]
        # print(f"pe shape: {pe.shape}")
        pe = pe.expand_as(x)
        # print(f"Expanded pe shape: {pe.shape}")
        x = x + pe
        return x
    


class RotaryEmbedding_HyperMixing(nn.Module):
    def __init__(self, device, key_size: int, rescaling_factor: Optional[float] = None):
        super(RotaryEmbedding_HyperMixing, self).__init__()
        UPPER_FREQ = 10000.0

        if rescaling_factor is None:
            self._inv_freq = 1.0 / (UPPER_FREQ ** (torch.arange(0, key_size, 2, device=device).float() / key_size))
        else:
            updated_base = UPPER_FREQ * (rescaling_factor ** (key_size / (key_size - 2)))
            self._inv_freq = 1.0 / (updated_base ** (torch.arange(0, key_size, 2, device=device).float() / key_size))

    def _compute_cos_sin_tables(self, device, heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = heads.size(1)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self._inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(heads.dtype)

        # cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        cos_cached = emb.cos().unsqueeze(0)
        # sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0)

        return cos_cached, sin_cached

    def _apply_rotary_pos_emb(self, heads: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_size = heads.size(-1)
        x1 = heads[..., : head_size // 2]
        x2 = heads[..., head_size // 2 :]
        heads_rotated = torch.cat((-x2, x1), dim=-1)
        
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
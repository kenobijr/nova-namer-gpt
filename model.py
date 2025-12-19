"""
NovaNamerGPT Transformer Architecture

This module implements a character-level GPT transformer optimized for name generation.
Features multi-head attention, feed-forward networks, and layer normalization with
proper weight initialization following GPT-2 standards.

Classes:
    GPTconfig: Model architecture hyperparameters
    MultiHeadAttention: Scaled dot-product attention with causal masking
    Ffw: Feed-forward network with ReLU activation
    TransformerBlock: Single transformer layer with attention and feed-forward
    GPT: Complete transformer model with embeddings and language modeling head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTconfig:
    """
    model architecture configuration for transformer
    - defines layer dimensions and attention parameters
    - controls dropout rates and bias settings
    - sets context length and vocabulary size
    """

    # model dimensions
    context_len: int = 64  # maximum sequence length (block size)
    vocab_size: int = 61  # character vocabulary size
    n_embd: int = 256  # embedding dimension
    n_head: int = 8  # number of attention heads
    kv_head: int = 8  # number of kv heads
    n_layer: int = 8  # number of transformer blocks
    # regularization
    dropout: float = 0.2
    # feed-forward network
    ffw_widen: int = 4  # expansion factor for ffn hidden layer
    # bias settings for different components
    a_bias: bool = True  # attention projection bias
    ffw_bias: bool = True  # feed-forward layer bias
    lm_head_bias: bool = False  # language modeling head bias


class MultiHeadAttention(nn.Module):
    """
    multi-head scaled dot-product attention with causal masking
    - computes attention weights for all heads in parallel
    - applies triangular mask to prevent looking at future tokens
    - includes dropout for regularization
    - relation kv_head & n_head dictate attention mode
    If kv_head == n_head -> multi-head attention
    If kv_head == 1, attention is multi-query attention
    Anyother valid kv_head will be grouped-query attention
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            "Ratio n_embd / n_head must have no remainder."
        )
        self.n_head = config.n_head
        self.head_size: int = config.n_embd // config.n_head
        self.kv_head = config.kv_head
        self.n_rep = self.n_head // config.kv_head  # kv repetition factor
        assert self.n_rep * config.kv_head == self.n_head, (
            "n_head must be divisible by kv_head."
        )
        self.query = nn.Linear(
            config.n_embd, self.n_head * self.head_size, bias=config.a_bias
        )
        self.key = nn.Linear(
            config.n_embd, self.kv_head * self.head_size, bias=config.a_bias
        )
        self.value = nn.Linear(
            config.n_embd, self.kv_head * self.head_size, bias=config.a_bias
        )

        # output projection to combine all heads
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.a_bias)
        self.dropout = nn.Dropout(config.dropout)

        # causal mask - lower triangular matrix prevents future token access
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.context_len, config.context_len)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels

        # compute q, k, v projections
        q, k, v = self.query(x), self.key(x), self.value(x)

        # split q into unique query heads: (B, T, n_head, head_size)
        q = q.view(B, T, self.n_head, self.head_size)
        # split kv into kv heads: (B, T, kv_head, head_size)
        k = k.view(B, T, self.kv_head, self.head_size)
        v = v.view(B, T, self.kv_head, self.head_size)

        # copy kv n_rep times to share it for number of heads
        k = torch.repeat_interleave(k, self.n_rep, dim=2)
        v = torch.repeat_interleave(v, self.n_rep, dim=2)

        # transpose to (B, n_head, T, head_size)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # scaled dot-product attention
        att = (
            q @ torch.transpose(k, dim0=-1, dim1=-2)
        ) * self.head_size**-0.5  # scale: sqrt(d_k)

        # apply causal mask and softmax
        att = F.softmax(att.masked_fill(self.tril[:T, :T] == 0, float("-inf")), dim=-1)

        # apply attention to values and reshape: (B, n_head, T, head_size)
        out = (self.dropout(att) @ v).transpose(1, 2).reshape(B, T, C)

        return self.dropout(self.proj(out))


class Ffw(nn.Module):
    """
    feed-forward network with relu activation
    - expands to wider hidden dimension then projects back
    - includes dropout for regularization
    - uses explicit layer naming for weight initialization
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        hidden_dim = config.n_embd * config.ffw_widen  # expand by widening factor
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.ffw_bias)  # expand
        self.relu = nn.ReLU()
        self.proj = nn.Linear(
            hidden_dim, config.n_embd, bias=config.ffw_bias
        )  # project back
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    single transformer block with pre-layer normalization
    - attention for communication between positions
    - feed-forward for computation within positions
    - residual connections around both sublayers
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.multi_head_sa = MultiHeadAttention(config)
        self.ffw = Ffw(config)
        self.rmsn1 = nn.RMSNorm(config.n_embd)  # pre-attention norm
        self.rmsn2 = nn.RMSNorm(config.n_embd)  # pre-ffn norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm architecture: norm -> sublayer -> residual
        x = x + self.multi_head_sa((self.rmsn1(x)))  # attention with residual
        x = x + self.ffw(self.rmsn2(x))  # feed-forward with residual
        return x


class GPT(nn.Module):
    """
    complete gpt transformer model for character-level language modeling
    - token and position embeddings
    - stack of transformer blocks
    - language modeling head with weight tying
    - proper weight initialization following gpt-2 standards
    """

    def __init__(self, config: GPTconfig, init_weights: bool = True):
        super().__init__()
        assert isinstance(config, GPTconfig), "Invalid config type."
        assert config.n_head > 0, "n_head must be positive."
        assert config.n_layer > 0, "n_layer must be positive."
        assert config.vocab_size > 0, "vocab_size must be positive."
        self.config = config
        # embeddings & transformer blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(
                    config.context_len, config.n_embd
                ),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                rmsn_f=nn.RMSNorm(config.n_embd),  # final layer norm
            )
        )
        # language modeling head
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.lm_head_bias
        )
        # weight tying - share parameters between token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight
        # initialize weights if requested
        if init_weights:
            self.apply(self._init_weights)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """
        forward pass through transformer
        - if targets provided: returns logits and cross-entropy loss
        - if no targets: returns logits for next token prediction only
        """
        B, T = idx.shape
        assert T <= self.config.context_len, (
            f"T: {T} exceeds context_len {self.config.context_len}"
        )
        # create position indices for current sequence
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        # token and position embeddings
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        # forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.rmsn_f(x)
        # compute logits and loss
        if targets is not None:
            # training mode - compute loss over all positions
            logits = self.lm_head(x)
            # flatten (B,T,C) -> (B*T,C) & (B,T) -> (B*T) for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference mode - only compute logits for last position / timestep
            logits = self.lm_head(x[:, [-1], :])  # list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        """
        init weights following gpt-2 standards
        - normal distribution for linear and embedding layers
        - scaled initialization for residual projections
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # standard init with small normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # scaled init for residual projections
        for name, param in self.named_parameters():
            if name.endswith(
                "proj.weight"
            ):  # catches both attention and ffw projections
                # scale by depth for proper gradient flow
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

    def get_num_params(self, none_embedding=True):
        """
        count total model parameters
        - non_embedding=True excludes positional embeddings from count
        - token embeddings included due to weight tying with output layer
        """
        n_params = sum(p.numel() for p in self.parameters())
        if none_embedding:
            n_params -= (
                self.transformer.wpe.weight.numel()
            )  # subtract position embeddings
        return n_params

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tiny decoder-only transformer for SLM meta-optimizer inner task.
Pure PyTorch, no transformers dependency.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


# Fixed character vocab for reproducible SLM tasks (subset of printable ASCII)
DEFAULT_CHARS = (
    " \n\t"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()"
)
DEFAULT_VOCAB_SIZE = len(DEFAULT_CHARS)


def build_vocab(chars: str = DEFAULT_CHARS) -> Tuple[dict, dict]:
    """Return char2idx and idx2char dicts."""
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode_corpus(text: str, char2idx: dict, default_idx: int = 0) -> torch.Tensor:
    """Encode string to long tensor of token ids. Unknown chars map to default_idx."""
    ids = [char2idx.get(c, default_idx) for c in text]
    return torch.tensor(ids, dtype=torch.long)


def get_corpus_tensor(
    text: str,
    char2idx: dict,
    device: torch.device,
) -> torch.Tensor:
    """Return 1D long tensor of token ids on device."""
    t = encode_corpus(text, char2idx)
    return t.to(device)


def sample_batch_slm(
    corpus_ids: torch.Tensor,
    batch_size: int,
    context_len: int,
    step: int,
    data_seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample batch_size contiguous chunks from corpus for next-token prediction.
    Returns: input_ids [B, context_len], target_ids [B, context_len] (target = input shifted by 1).
    """
    L = corpus_ids.size(0)
    if L <= context_len + 1:
        raise ValueError("Corpus too short for context_len")
    max_start = L - context_len - 1
    g = torch.Generator(device=device)
    g.manual_seed(data_seed + step)
    starts = torch.randint(0, max_start, (batch_size,), device=device, generator=g)
    inputs = []
    targets = []
    for b in range(batch_size):
        s = int(starts[b].item())
        chunk = corpus_ids[s : s + context_len + 1]
        inputs.append(chunk[:context_len])
        targets.append(chunk[1 : context_len + 1])
    return torch.stack(inputs), torch.stack(targets)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """Decoder-only transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
    ):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(context_len, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, context_len) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]; clamp to valid range in case of encoding drift
        idx = idx.clamp(0, self.vocab_size - 1)
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

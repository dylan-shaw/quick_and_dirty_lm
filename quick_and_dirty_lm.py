#!/usr/bin/env python3

from pathlib import Path
from tarfile import TarFile, TarInfo
import math
from collections import deque

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import numpy as np

class MultiheadAttention(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 heads: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        assert out_features % heads == 0
        self.__qkv = nn.Linear(in_features, out_features * 3, bias=bias)
        self.__out = nn.Linear(out_features, out_features, bias=False)
        self.__heads = heads
        self.__head_dim = out_features // heads
        self.__dropout = dropout
        self.__out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv: Tensor = self.__qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.__heads, self.__head_dim).transpose(1, 2)
        k = k.view(B, T, self.__heads, self.__head_dim).transpose(1, 2)
        v = v.view(B, T, self.__heads, self.__head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.__dropout if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.__out_features)
        y = self.__out(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 heads: int,
                 dropout: float,
                 qkv_bias: bool):
        super().__init__()
        self.__norm1 = nn.LayerNorm(in_features)
        self.__attention = MultiheadAttention(in_features, out_features, heads, dropout, qkv_bias)
        self.__dropout = nn.Dropout(dropout)
        self.__norm2 = nn.LayerNorm(out_features)
        self.__linear = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.__norm1(x)
        x = self.__attention(x)
        x = self.__dropout(x)

        if x.shape == skip.shape:
            # we can only really do this if the number
            # of input features is the same as the number
            # of output features.
            x = x + skip

        skip = x
        x = self.__norm2(x)
        x = self.__linear(x)
        x = self.__dropout(x)
        x = x + skip

        return x

class GPTModel(nn.Module):
    """
    This is a composition of transformer blocks that mimics GPT-2.
    """
    def __init__(self, vocab_size: int, embedding: int, context_length: int, heads: int, dropout: float, qkv_bias: bool, layers: int):
        super().__init__()
        self.__token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding)
        self.__position_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding)
        blocks = [TransformerBlock(embedding, embedding, heads, dropout, qkv_bias) for _ in range(layers)]
        self.__layers = nn.Sequential(*blocks)
        self.__norm = nn.LayerNorm(embedding)
        self.__last = nn.Sequential(
            nn.Linear(embedding, vocab_size, bias=False)
        )
        self.__context_length = context_length

    def forward(self, tokens: Tensor) -> Tensor:
        token_e = self.__token_embedding(tokens)
        pos_e = self.__position_embedding(torch.arange(self.__context_length, device=tokens.device))
        x = token_e + pos_e
        x = self.__layers(x)
        x = self.__norm(x)
        logits = self.__last(x)
        return logits

class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, filename: Path, block_length: int, dtype: str):
        self.__file = np.memmap(filename, getattr(np, dtype), mode='r')
        self.__num_samples = self.__file.size // block_length
        assert self.__file.size % block_length == 0
        self.__block_length = block_length

    def __len__(self):
        return self.__num_samples

    def __getitem__(self, index: int) -> Tensor:
        offset = index * self.__block_length
        x = self.__file[offset:offset + self.__block_length]
        return torch.from_numpy(x)

class Trainer:
    def __init__(self,
                 net: torch.nn.Module,
                 max_learning_rate: float = 1.0e-3,
                 optimizer_name: str = 'Adam',
                 steps_per_epoch: int = 1000,
                 num_epochs: int = 100,
                 warmup_epochs: int = 10,
                 device: str | None = None):
        self.__net = net
        self.__train_step = 0
        self.__steps_per_epoch = steps_per_epoch
        self.__num_epochs = num_epochs
        self.__train_loss = deque(maxlen=steps_per_epoch)
        self.__device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.__optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer_name)(
            net.parameters(),
            max_learning_rate
        )
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.__optimizer,
            start_factor=1.0e-3
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.__optimizer,
            T_max=(steps_per_epoch - warmup_epochs)
        )
        self.__lr_sched = torch.optim.lr_scheduler.SequentialLR(
            self.__optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
        self.__grad_scaler = torch.amp.grad_scaler.GradScaler(self.__device)

    def epoch(self) -> int:
        return self.__train_step // self.__steps_per_epoch

    def done(self) -> bool:
        return self.__train_step >= (self.__num_epochs * self.__steps_per_epoch)

    def iterate_training(self, x: torch.Tensor, y: torch.Tensor) -> float | None:
        assert x.dtype == torch.uint16
        assert y.dtype == torch.uint16
        with torch.autocast(self.__device, dtype=torch.float16):
            y_predicted: Tensor = self.__net(x.long())
            assert y_predicted.dtype == torch.float16
            logits = y_predicted.view(-1, y_predicted.size(-1))
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets.long())
            assert loss.dtype == torch.float32
        self.__grad_scaler.scale(loss).backward()
        self.__grad_scaler.step(self.__optimizer)
        self.__grad_scaler.update()
        self.__optimizer.zero_grad()
        self.__train_loss.append(loss.item())
        self.__train_step += 1
        if self.__train_step % self.__steps_per_epoch == 0:
            loss = sum(self.__train_loss) / len(self.__train_loss)
            self.__lr_sched.step()
            return loss
        return None

    def validate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        assert x.dtype == torch.uint16
        assert y.dtype == torch.uint16
        y_predicted: Tensor = self.__net(x.long())
        logits = y_predicted.view(-1, y_predicted.size(-1))
        targets = y.view(-1)
        loss = F.cross_entropy(logits, targets.long())
        return loss.item()
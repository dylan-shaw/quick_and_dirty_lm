"""
Contains various functions and classes for training and testing language models.
"""
# SPDX-License-Identifier: Apache-2.0.

from abc import ABC, abstractmethod
from pathlib import Path
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
        self.__input_norm = nn.LayerNorm(in_features)
        self.__attention = MultiheadAttention(
            in_features,
            out_features,
            heads,
            dropout,
            qkv_bias
        )
        self.__output_norm = nn.LayerNorm(out_features)
        self.__linear = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.ReLU(),
            nn.Linear(out_features * 4, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.__attention(x) + x
        x = self.__input_norm(x)
        x = self.__linear(x) + x
        x = self.__output_norm(x)
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
            nn.Linear(embedding, vocab_size)
        )
        self.__context_length = context_length

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.size(dim=1)
        if sequence_length > self.__context_length:
            # use only the last tokens in the token array,
            # since our position embeddings only go up to
            # the max context length
            tokens = tokens[: -self.__context_length]

        token_e = self.__token_embedding(tokens)
        pos_e = self.__position_embedding(torch.arange(sequence_length, device=tokens.device))
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
                 max_learning_rate: float = 1.0e-4,
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

class Tokenizer(ABC):
    """
    This class is for encoding or decoding tokens.
    There are many ways to do this, so this class may be derived
    for which ever algorithm or library being used for tokenization.
    """
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes characters as a sequence, represented by a list of integers.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a set of tokens into text.
        """
        raise NotImplementedError()

class SentencePieceTokenizer(Tokenizer):
    """
    An implementation of the tokenizer class using SentencePiece.
    <br>
    <br>
    *Note: This requires the `sentencepiece` package.*
    """
    def __init__(self, model_filename: str):
        """
        Creates a SentencePiece tokenizer.

        :param model_filename: The path to the tokenizer model file.
        """
        import sentencepiece as sp
        self.__model = sp.SentencePieceProcessor()
        self.__model.LoadFromFile(model_filename)

    def encode(self, text: str) -> list[int]:
        return self.__model.Encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.__model.Decode(tokens)

class Generator:
    """
    Used for generating text from a model.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer: Tokenizer,
                 device: torch.device,
                 sample: bool = True,
                 top_k: int = 10,
                 seed: int = 0):
        self.__model = model
        self.__tokenizer = tokenizer
        self.__device = device
        self.__rng = torch.Generator(device=device)
        self.__rng.manual_seed(seed)
        self.__sample = sample
        self.__top_k = top_k

    def generate(self, text: str, max_tokens: int) -> str:
        tokens = torch.tensor(self.__tokenizer.encode(text), dtype=torch.long).unsqueeze(dim=0)
        tokens = tokens.to(self.__device)
        with torch.no_grad():
            for _ in range(max_tokens):
                logits: torch.Tensor = self.__model(tokens)
                if self.__sample:
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, self.__top_k, dim=-1)
                    next_token = torch.multinomial(top_k_probs, num_samples=1, generator=self.__rng)
                    next_token = top_k_indices.gather(1, next_token)
                else:
                    next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(dim=0)
                tokens = torch.concat((tokens, next_token), dim=1)
        output_text = self.__tokenizer.decode(tokens.flatten().cpu().tolist())
        return output_text
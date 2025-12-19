#!/usr/bin/env python3

from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """
    This represents a causal single-head attention layer.
    Normally, you want more than one head in a transformer, so this is meant to be stacked.

    :param in_features: The number of input features per token.
    :param out_features: The number of output features per token.
    :param context_length: The context length, used to construct the causal mask.
    :param dropout: The dropout probability, used on the causal mask.
    :param bias: Whether to use bias on the key, value and query transforms.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 context_length: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        self.__key = nn.Linear(in_features, out_features, bias=bias)
        self.__query = nn.Linear(in_features, out_features, bias=bias)
        self.__value = nn.Linear(in_features, out_features, bias=bias)
        self.__dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x: Tensor) -> Tensor:
        mask: Tensor = self.mask # a registered buffer
        k: Tensor = self.__key(x)
        q: Tensor = self.__query(x)
        v: Tensor = self.__value(x)
        if len(x.shape) == 2:
            # When there is no batch dimension, it's a simple matrix transpose
            scores = q @ k.T
        else:
            # When there is a batch dimension, it's a more general transpose
            scores = q @ k.transpose(1, 2)
        scores.masked_fill_(mask, -torch.inf)
        weights = torch.softmax(scores / (k.shape[-1] ** 0.5), dim=-1)
        weights = self.__dropout(weights)
        y = weights @ v
        return y

class MultiheadAttention(nn.Module):
    """
    This module is just a stacked set of single head attention layers.
    There are slightly more efficient ways to implement this, but this version is easier to read and write.

    :param in_features: The number of input features per token.
    :param out_features: The number of output features per token, per head.
    :param heads: The number of attention heads.
    :param context_length: The number of tokens in a single batch.
    :param dropout: The dropout probability, used on the causal mask.
    :param bias: Whether or not to use bias in the key, value, and query transforms.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 heads: int,
                 context_length: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        self.__heads = nn.ModuleList(
            SelfAttention(in_features, out_features // heads, context_length, dropout, bias) for _ in range(heads)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.concat([head(x) for head in self.__heads], dim=-1)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 context_length: int,
                 heads: int,
                 dropout: float,
                 qkv_bias: bool):
        super().__init__()
        self.__norm1 = nn.LayerNorm(in_features)
        self.__attention = MultiheadAttention(in_features, out_features, heads, context_length, dropout, qkv_bias)
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
        layers = [TransformerBlock(embedding, embedding, context_length, heads, dropout, qkv_bias) for _ in range(layers)]
        self.__layers = nn.Sequential(*layers)
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
    """
    This is a dataset for pretraining a model.
    It expects a directory of pre-tokenized files and uses them as training samples.
    In general, each file is expected to be at least the size of the context window.
    You can also use this for training character-level language models, just set the
    data type to be uint8 and make sure your vocabulary size fits all the characters.

    :param directory: The directory where the text files live. This directory is searched recursively for files.
    :param context_length: The context length for the model being trained. Files in the data directory must be at least one token greater in size than the context size.
    """
    def __init__(self, directory: Path, context_length: int, dtype: torch.dtype):
        self.__filenames: list[Path] = []
        self.__context_length = context_length
        self.__dtype = dtype
        dir_queue = [ directory ]
        while len(dir_queue) > 0:
            dir = dir_queue.pop()
            for entry in dir.glob('*'):
                if entry.is_dir():
                    dir_queue.append(entry)
                    continue
                size = entry.stat().st_size
                if size < ((self.__context_length + 1) * dtype.itemsize):
                    # the file has to be larger than the context size by one token
                    continue
                self.__filenames.append(entry)

    def __len__(self):
        return len(self.__filenames)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        filename = str(self.__filenames[index])
        xy = torch.from_file(filename, shared=False, size=(self.__context_length + 1), dtype=self.__dtype)
        return xy[0:self.__context_length], xy[1:self.__context_length + 1]

def pretrain(training_directory: Path,
             validate_directory: Path,
             context_length: int,
             dtype: torch.dtype,
             batch_size: int,
             epochs: int,
             optimizer_name: str,
             max_learning_rate: float,
             warmup_epochs: int,
             vocab_size: int,
             layers: int,
             heads: int,
             embedding_dim: int):
    training_data = PretrainingDataset(training_directory, context_length, dtype=dtype)
    validate_data = PretrainingDataset(validate_directory, context_length, dtype=dtype)
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=1, shuffle=False)
    device = torch.device('cuda')
    model = GPTModel(vocab_size=vocab_size,
                     embedding=embedding_dim,
                     context_length=context_length,
                     heads=heads,
                     dropout=0.1,
                     qkv_bias=False,
                     layers=layers)
    model = model.to(device)
    optimizer_type = getattr(torch.optim, optimizer_name)
    optimizer: torch.optim.Optimizer = optimizer_type(model.parameters(), max_learning_rate)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                     start_factor=1.0e-5,
                                                     end_factor=max_learning_rate)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs))
    lr_sched = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                     schedulers=[warmup_sched, cosine_sched],
                                                     milestones=[warmup_epochs])
    for epoch in range(epochs):
        training_loss = 0.0
        for sample in training_loader:
            x, y = sample
            x: Tensor = x.to(device).long()
            y: Tensor = y.to(device).long()
            y_predicted: Tensor = model(x)
            logits = y_predicted.view(-1, y_predicted.size(-1))
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(training_loader)

        with torch.no_grad():
            validate_loss = 0.0
            for sample in validate_loader:
                x, y = sample
                x: Tensor = x.to(device).long()
                y: Tensor = y.to(device).long()
                y_predicted: Tensor = model(x)
                logits = y_predicted.view(-1, y_predicted.size(-1))
                targets = y.view(-1)
                loss = F.cross_entropy(logits, targets)
                validate_loss += loss.item()
            validate_loss /= len(validate_loader)
        lr_sched.step()

        print(f'[{epoch:04}]: train_loss:{training_loss:.04f} val_loss:{validate_loss:.04f}')

def _pretrain(args):
    pretrain(training_directory=Path(args.training_directory),
             validate_directory=Path(args.validate_directory),
             context_length=args.context_length,
             dtype=getattr(torch, args.dtype),
             batch_size=args.batch_size,
             epochs=args.epochs,
             optimizer_name=args.optimizer,
             max_learning_rate=args.max_learning_rate,
             warmup_epochs=args.warmup_epochs,
             vocab_size=args.vocab_size,
             layers=args.layers,
             heads=args.heads,
             embedding_dim=args.embedding_dim)

def _smoke_test_gpt(args):

    print('smoke testing gpt model')
    model = GPTModel(vocab_size=10,
                     embedding=16,
                     context_length=12,
                     heads=2,
                     dropout=0.1,
                     qkv_bias=False,
                     layers=2)

    print('testing without batch dimension')
    x = torch.tensor([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6
    ]).long()
    y: Tensor = model(x)
    print('  done')
    print('testing with batch dimension')
    x_batch = torch.stack((x, x), dim=0)
    y = model(x_batch)
    print('  done')
    print('seems good to go')

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers()

    smoke_test_gpt = subparsers.add_parser('smoke-test-gpt')
    smoke_test_gpt.set_defaults(func=_smoke_test_gpt)

    pretrain_parser = subparsers.add_parser('pretrain')
    pretrain_parser.add_argument('--training-directory', type=str, default='test_data/training', help='The directory containing the training data.')
    pretrain_parser.add_argument('--validate-directory', type=str, default='test_data/validate', help='The directory containing the validation data.')
    pretrain_parser.add_argument('--context-length', type=int, default=128, help='The context length to train the model at.')
    pretrain_parser.add_argument('--dtype', type=str, default='uint8', help='The data type of a single token.')
    pretrain_parser.add_argument('--batch-size', type=int, default=16, help='The batch size for the training data.')
    pretrain_parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for.')
    pretrain_parser.add_argument('--optimizer', type=str, default='Adam', help='The optimizer to train the model with.')
    pretrain_parser.add_argument('--max-learning-rate', type=float, default=1.0e-2, help='The peak learning rate to reach after warmup')
    pretrain_parser.add_argument('--warmup-epochs', type=int, default=4, help='The number of epochs to spend winding up LR.')
    pretrain_parser.add_argument('--vocab-size', type=int, default=256, help='The size of the vocabulary')
    pretrain_parser.add_argument('--layers', type=int, default=4, help='The number of transformers to put into the model')
    pretrain_parser.add_argument('--heads', type=int, default=4, help='The number of attention heads per transformer.')
    pretrain_parser.add_argument('--embedding-dim', type=int, default=128, help='The number of dimensions in an embedding.')
    pretrain_parser.set_defaults(func=_pretrain)

    args = parser.parse_args()
    args.func(args)
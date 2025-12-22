"""
This module will train a tokenizer on a subset of the training dataset.
The tokenizer and all other intermediate files are generated in the example
directory.
"""

# You'll need to install the following dependencies to run this script:
#   sentencepiece datasets

from pathlib import Path
from random import Random
from array import array

import sentencepiece as sp

from datasets import (
    load_dataset,
    DatasetDict,
    Dataset
)

def load_datasets() -> tuple[Dataset, Dataset]:
    ds: DatasetDict = load_dataset("dylan-shaw/gutenberg_8k")
    return ds['train'], ds['validation']

def dataset_to_string(dataset: Dataset, max_size: int = 1_000_000_000, shuffle_seed: int = 0) -> str:
    result = ''
    indices = list(range(0, len(dataset) - 1))
    Random(shuffle_seed).shuffle(indices)
    for index in indices:
        entry = dataset[index]
        assert isinstance(entry, dict)
        result += entry['txt']
        if len(result) >= max_size:
            break
    return result

def tokenize_dataset(processor: sp.SentencePieceProcessor, dataset: Dataset, output_filename: Path, sample_size: int):
    result = bytearray()
    for entry in dataset:
        assert isinstance(entry, dict)
        txt = entry['txt']
        tokens: list[int] = processor.Encode(txt, out_type=int)
        if len(tokens) < sample_size:
            continue
        tokens_subset = tokens[:sample_size]
        # create a 16-bit array of these tokens
        buf = array('H')
        buf.extend(tokens_subset)
        result.extend(buf.tobytes())
    with open(output_filename, 'wb') as f:
        f.write(result)

def main():
    train_ds, val_ds = load_datasets()
    model_prefix = 'examples/gpt/tokenizer'
    model_filename = Path(f'{model_prefix}.model')
    # train the model if it does not exist
    if not model_filename.exists():
        corpus = dataset_to_string(train_ds)
        corpus_filename = Path('examples/gpt/corpus.txt')
        corpus_filename.unlink(missing_ok=True)
        with open(corpus_filename, 'w') as f:
            f.write(corpus)
        vocab_size = 512
        sp.SentencePieceTrainer.Train(f'--input={corpus_filename} --vocab_size={vocab_size} --model_prefix={model_prefix}')
        # cleanup
        corpus_filename.unlink()
    # tokenize the datasets
    processor = sp.SentencePieceProcessor()
    processor.LoadFromFile(str(model_filename))
    sample_size = 1024
    tokenize_dataset(processor, train_ds, Path('examples/gpt/train.bin'), sample_size)
    tokenize_dataset(processor, val_ds, Path('examples/gpt/val.bin'), sample_size)


if __name__ == '__main__':
    main()
"""
This example is for training a foundation model from Gutenberg text.
"""

from pathlib import Path
import math

from quick_and_dirty_lm import (
    PretrainingDataset,
    Trainer,
    GPTModel
)

import torch

def main(context_length: int = 512):
    train_ds = PretrainingDataset(Path('examples/gpt/train.bin'), block_length=1024, dtype='uint16')
    val_ds = PretrainingDataset(Path('examples/gpt/val.bin'), block_length=1024, dtype='uint16')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    net = GPTModel(
        vocab_size=512,
        embedding=1024,
        context_length=context_length,
        heads=8,
        dropout=0.01,
        qkv_bias=False,
        layers=12
    )
    net.to(device)
    steps_per_epoch = 1000
    total_epochs = len(train_loader) // steps_per_epoch
    trainer = Trainer(
        net,
        max_learning_rate=1.0e-4,
        steps_per_epoch=steps_per_epoch,
        num_epochs=total_epochs,
        device=device_name
    )
    best_val_loss = math.inf
    while not trainer.done():
        net.train()
        for train_sample in train_loader:
            x: torch.Tensor = train_sample[:, 0:context_length]
            y: torch.Tensor = train_sample[:, 1:context_length+1]
            train_loss: float | None = trainer.iterate_training(x.to(device), y.to(device))
            if train_loss is None:
                continue
            epoch = trainer.epoch()
            with torch.no_grad():
                net.eval()
                validate_loss = 0.0
                for val_sample in val_loader:
                    x: torch.Tensor = val_sample[:, 0:context_length]
                    y: torch.Tensor = val_sample[:, 1:context_length+1]
                    validate_loss += trainer.validate(x.to(device), y.to(device))
            validate_loss /= len(val_loader)
            if validate_loss < best_val_loss:
                best_val_loss = validate_loss
                torch.save(net.state_dict(), 'model.pt')
            print(f'{epoch}/{total_epochs}:')
            print(f'     train loss: {train_loss:.04f}')
            print(f'  validate loss: {validate_loss:.04f}')
        if trainer.done():
            break

if __name__ == '__main__':
    main()
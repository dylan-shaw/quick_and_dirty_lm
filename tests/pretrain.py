import quick_and_dirty_lm as lm

import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def main():
    pass

if __name__ == '__main__':
    main()
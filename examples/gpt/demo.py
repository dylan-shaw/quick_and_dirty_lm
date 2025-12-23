from pathlib import Path

import torch

from language_model import (
    GPTModel,
    SentencePieceTokenizer,
    Generator
)

def main():
    model_filename = 'examples/gpt/model.pt'
    tokenizer_filename = 'examples/gpt/tokenizer.model'
    context_length = 256
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # Load the trained model
    net = GPTModel(
        vocab_size=512,
        embedding=384,
        context_length=context_length,
        heads=6,
        dropout=0.01,
        qkv_bias=False,
        layers=6
    )
    net.load_state_dict(torch.load(model_filename))
    net.to(device)

    # Load the tokenizer
    tokenizer = SentencePieceTokenizer(tokenizer_filename)
    generator = Generator(net, tokenizer, device)

    # Get user input for the initial text
    while True:
        initial_text = input('>')
        if initial_text == '':
            break
        result = generator.generate(initial_text, max_tokens=32)
        print(result)

if __name__ == '__main__':
    main()
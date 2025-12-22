About
=====

This is a library (and program) for training and working on language models.
It comes with the basics for making a GPT-like language model, but also allows
you to put your own stuff in it. It is not meant to contain every single technique
for training the language model, just the basics that can be used as a good starting
point. In many cases, it is all you really need to make a functioning language model.

### Getting Started

While you are free to copy the code, and even modify it, in your own repo, it can also
be used as-is without modifying it at all. To start, make sure you have the dependencies installed.
You can make a simple virtual environment to do that. On Linux systems (including WSL) that
would be done like this:

```
python3 -m venv .venv
. .venv/scripts/activate
pip3 install -r requirements.txt
```

Once that's done, you can run the example!
The first part of the example is downloading the dataset and training a tokenizer.
To do that, run the following script:

```
python3 examples/gpt/train_tokenizer.py
```

Once you have a tokenizer, you can start training a foundation model.
The example trains with a batch size of 1. While this can lead to some
noisy gradients, it also means that you only need a couple of gigs of VRAM
to train the model. To do this, run:

```
python3 examples/gpt/pretrain.py
```

There are also VS Code files for running this scripts from a GUI-based editor.

Happy hacking!

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

If you'd like to verify that you've got a working environment,
you can run the pretraining function on the test data, using a
small model that fits in most GPUs.

```
./quick_and_dirty_lm.py pretrain
```
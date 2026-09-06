# Transformer From Scratch

![transformer architecture](/images/architecture.png)

A small char-level language model, built by hand to actually understand how transformers work, not just how to call one.

It's a decoder-only transformer (GPT-style), implemented in NumPy and trained to predict the next character in a piece of text. No high-level deep learning framework doing the heavy lifting: attention, backpropagation, and training are all implemented from first principles.
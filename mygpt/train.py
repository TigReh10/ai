"""Train YOUR OWN AI from scratch.

This trains the character-level GPT defined in model.py on whatever text you put
in data/input.txt. The model starts knowing NOTHING and learns purely from your
text — you are training your own neural network.

Run (from inside the mygpt/ folder):
    python train.py

Tip: the more (and cleaner) text in data/input.txt, the better it learns.
"""
from __future__ import annotations

import json
import os

import torch

from model import GPT

# ---- hyperparameters (try changing these!) ----
batch_size = 32        # sequences processed in parallel
block_size = 128       # how many characters of context
max_iters = 3000       # training steps
eval_interval = 250    # how often to print the loss
learning_rate = 3e-4
eval_iters = 100
n_embd = 128           # size of the model's "thinking" vectors
n_head = 4             # attention heads
n_layer = 4            # transformer blocks (depth)
dropout = 0.1
# -----------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

DATA_PATH = os.path.join("data", "input.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Build the vocabulary from the unique characters in your text.
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(tokens):
    return "".join(itos[i] for i in tokens)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training on {device}. Vocab size: {vocab_size}. Parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "model.pt"))
    meta = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "config": {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "dropout": dropout,
        },
    }
    with open(os.path.join("checkpoints", "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("\nSaved your trained model to checkpoints/model.pt")

    print("\n--- A sample from your freshly trained AI ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))


if __name__ == "__main__":
    main()

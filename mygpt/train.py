"""Train YOUR OWN AI from scratch — now scalable.

This trains the character-level GPT defined in model.py on data/input.txt.
The model starts knowing NOTHING and learns purely from your text.

MAKE IT SMARTER: pick a bigger size with the MODEL_SIZE env var.
    MODEL_SIZE=small  python train.py    # default, fine on a CPU
    MODEL_SIZE=medium python train.py    # better; slow on CPU, fast on GPU
    MODEL_SIZE=large  python train.py    # needs a GPU
    MODEL_SIZE=xl     python train.py    # needs a strong GPU

Progress is saved to checkpoints/ periodically AND on Ctrl+C.
"""
from __future__ import annotations

import json
import os

import torch

from model import GPT

# ---- model size presets (bigger = smarter, but slower & needs more memory) ----
PRESETS = {
    "small":  dict(n_embd=128, n_head=4, n_layer=4, block_size=128, max_iters=3000,  batch_size=32),
    "medium": dict(n_embd=256, n_head=8, n_layer=6, block_size=256, max_iters=6000,  batch_size=48),
    "large":  dict(n_embd=384, n_head=6, n_layer=6, block_size=256, max_iters=8000,  batch_size=64),
    "xl":     dict(n_embd=512, n_head=8, n_layer=8, block_size=512, max_iters=12000, batch_size=64),
}
MODEL_SIZE = os.environ.get("MODEL_SIZE", "small").lower()
cfg = PRESETS.get(MODEL_SIZE, PRESETS["small"])

batch_size = cfg["batch_size"]
block_size = cfg["block_size"]
max_iters = cfg["max_iters"]
n_embd = cfg["n_embd"]
n_head = cfg["n_head"]
n_layer = cfg["n_layer"]

eval_interval = 250
learning_rate = 3e-4
eval_iters = 100
dropout = 0.1
# -------------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

DATA_PATH = os.path.join("data", "input.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

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


def save_checkpoint(model):
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


def main():
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Size preset: {MODEL_SIZE}. Training on {device}. Vocab: {vocab_size}. Parameters: {n_params/1e6:.2f}M")
    print("Tip: you can press Ctrl+C any time — progress is saved automatically.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    try:
        for it in range(max_iters + 1):
            if it % eval_interval == 0:
                losses = estimate_loss(model)
                print(f"step {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}  (saved)")
                save_checkpoint(model)
            xb, yb = get_batch("train")
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        print("\nInterrupted — saving what we have so far...")

    save_checkpoint(model)
    print("\nSaved your trained model to checkpoints/model.pt")

    print("\n--- A sample from your trained AI ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))


if __name__ == "__main__":
    main()

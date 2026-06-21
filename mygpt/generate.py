"""Generate text from YOUR trained model.

Run (from inside the mygpt/ folder, after training):
    python generate.py --prompt "Once upon a time" --tokens 500
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from model import GPT


def load():
    with open(os.path.join("checkpoints", "vocab.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    cfg = meta["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(
        cfg["vocab_size"],
        cfg["n_embd"],
        cfg["n_head"],
        cfg["n_layer"],
        cfg["block_size"],
        cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join("checkpoints", "model.pt"), map_location=device))
    model.eval()
    return model, stoi, itos, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    model, stoi, itos, device = load()
    ids = [stoi[c] for c in args.prompt if c in stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = "".join(itos[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

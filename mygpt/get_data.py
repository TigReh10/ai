"""Download a large public-domain training corpus into data/input.txt.

The starter text in data/input.txt is tiny, which makes your AI overfit. This
script replaces it with a big, high-quality dataset so your model learns much
better writing.

Run (from inside the mygpt/ folder):
    python get_data.py                # default: Tiny Shakespeare (~1 MB)
    python get_data.py shakespeare    # Shakespeare's plays (~1 MB)
    python get_data.py alice          # Alice's Adventures in Wonderland
    python get_data.py sherlock       # The Adventures of Sherlock Holmes

Then train as usual:
    python train.py
"""
from __future__ import annotations

import os
import sys
import urllib.request

# Each dataset lists a few mirror URLs; the script tries them in order.
DATASETS = {
    "shakespeare": [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt",
    ],
    "alice": [
        "https://www.gutenberg.org/files/11/11-0.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    ],
    "sherlock": [
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ],
}


def download(name: str) -> None:
    urls = DATASETS.get(name)
    if not urls:
        print(f"Unknown dataset '{name}'. Options: {', '.join(DATASETS)}")
        return

    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", "input.txt")

    for url in urls:
        try:
            print(f"Downloading '{name}' from:\n  {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                text = response.read().decode("utf-8", errors="ignore")
            with open(dest, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\nSaved {len(text) / 1024:.0f} KB to {dest}")
            print("Now run:  python train.py")
            return
        except Exception as exc:
            print(f"  failed: {exc}\n")

    print(
        "All sources failed. Check your internet connection, or just paste your "
        "own text into data/input.txt manually."
    )


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "shakespeare"
    download(dataset)

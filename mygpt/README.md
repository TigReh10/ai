# MyGPT — build your OWN AI from scratch 🧠⚙️

This is a **real language model built from scratch** — the same kind of
transformer architecture that powers GPT/ChatGPT, just small enough that **you**
can build it and **train it yourself** on a normal computer (or for free in the
browser with GitHub Codespaces).

Nothing here is pre-trained. The model starts knowing *nothing* and learns only
from the text you give it. It is genuinely **your own AI**.

## What's inside

| File | What it is |
|------|-----------|
| `model.py` | The neural network, built by hand: embeddings, self-attention, transformer blocks |
| `train.py` | Trains the model on `data/input.txt` and saves it |
| `generate.py` | Generates new text from your trained model |
| `get_data.py` | Downloads a big public-domain dataset to train on |
| `data/input.txt` | Your training text (replace it with your own!) |

## Quick start

```bash
cd mygpt
pip install -r requirements.txt

# 1) Get a big, high-quality training set (~1 MB) so it learns well:
python get_data.py            # Tiny Shakespeare (great results!)
# other options: python get_data.py alice   /   python get_data.py sherlock

# 2) Train your AI:
python train.py

# 3) Generate text from YOUR model:
python generate.py --prompt "ROMEO:" --tokens 500
```

Training prints the loss going down — that is your AI *learning*. When it
finishes, it saves to `checkpoints/` and prints a writing sample.

## Use your OWN data instead

Want it to write like *you*? Replace `data/input.txt` with your own text — your
notes, a favourite book, song lyrics, code, anything. Aim for at least ~100 KB.
Then run `python train.py` again.

## Make it bigger / smarter

Open `train.py` and increase these (needs more time / a GPU):

```python
n_embd = 256     # wider "thinking" vectors
n_head = 8       # more attention heads
n_layer = 6      # deeper network
block_size = 256 # longer memory
max_iters = 6000 # train longer
```

More data + a bigger model + longer training = noticeably better writing.

## Honest expectations 💯

- This is the **real foundation** of how ChatGPT works — you are building and
  training an actual transformer, not faking it.
- But a model you train on a laptop / free cloud is *tiny* compared to ChatGPT
  (which has billions of parameters and trained on huge clusters for months). So
  it won't chat like ChatGPT — it learns to **write in the style of your text**,
  character by character.
- That's the honest trade: you fully **own and understand** every line of this
  AI, even if it's small.

This is the legit way to "make your own AI." Train it, tweak it, and watch it
learn. 🚀

# MyGPT — build your OWN AI from scratch 🧠⚙️

This is a **real language model built from scratch** — the same kind of
transformer architecture that powers GPT/ChatGPT, just small enough that **you**
can build it and **train it yourself** on a normal computer.

Nothing here is pre-trained. The model starts knowing *nothing* and learns only
from the text you give it. It is genuinely **your own AI**.

## What's inside

| File | What it is |
|------|-----------|
| `model.py` | The neural network, built by hand: embeddings, self-attention, transformer blocks |
| `train.py` | Trains the model on `data/input.txt` and saves it |
| `generate.py` | Generates new text from your trained model |
| `data/input.txt` | Your training text (replace it with your own!) |

## How to use it

```bash
cd mygpt
pip install torch            # (already in the project requirements)

# 1) Put your text in data/input.txt (a book, your notes, lyrics, code...)
# 2) Train your AI:
python train.py

# 3) Generate text from YOUR model:
python generate.py --prompt "Once upon a time" --tokens 500
```

Training prints the loss going down — that is your AI *learning*. When it
finishes, it saves to `checkpoints/` and prints a writing sample.

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
- But a model you train on a laptop is *tiny* compared to ChatGPT (which has
  billions of parameters and was trained on huge clusters for months). So it
  won't chat like ChatGPT — it learns to **write in the style of your text**,
  character by character.
- That's the honest trade: you fully **own and understand** every line of this
  AI, even if it's small. That understanding is worth far more than a wrapper
  around someone else's model.

This is the legit way to "make your own AI." Train it, tweak it, and watch it
learn. 🚀

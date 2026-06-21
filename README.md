# SmartAI 🧠🔍

A **self-hosted** AI assistant that runs its **own local LLM** — no external API,
no API key, no per-message cost. The model lives on *your* machine.

SmartAI is also **agentic**: it reasons about your question and, when it needs
fresh or factual information, it **searches the live web and reads pages on its
own** before answering — then cites its sources.

## Why this is different

- 🤖 **Its own LLM** — uses an open-source instruct model (Qwen / Llama, etc.)
  downloaded from Hugging Face and run **entirely locally**. No OpenAI, no keys.
- 🔍 **Live web search** via DuckDuckGo (free, no API key).
- 📄 **Page reading** — fetches and extracts readable text from any URL.
- 🔁 **ReAct loop** — the model plans, searches, reads, then answers with sources.
- 💬 **Two interfaces** — a command-line chat and a clean web UI.

## How it works

```
You ─► Local LLM (on your machine) ─► decides what to do next
                 │
                 ├► SEARCH: query   ─► live web results
                 ├► READ: url       ─► page text
                 └► ANSWER: ...      ─► grounded answer with sources
```

## Requirements

- Python 3.9+
- ~3–6 GB free disk for model weights (depends on the model you pick)
- A reasonably modern CPU works; a GPU makes it much faster

> Tip: start with the small default model. If you have a GPU or lots of RAM,
> switch to a bigger model in `.env` for smarter answers.

## Setup

```bash
git clone https://github.com/TigReh10/ai.git
cd ai
pip install -r requirements.txt
```

(Optional) choose a model — the default works out of the box:

```bash
cp .env.example .env   # then edit AI_MODEL if you want a different one
```

The first run downloads the model weights automatically. After that it runs
offline (web search aside).

## Usage

### Command line

```bash
python cli.py
```

```
You: who won the latest F1 race?
  [web_search] latest F1 race winner
  [read_url] https://...

SmartAI: ... (a fresh, sourced answer)
```

### Web app

```bash
python web_app.py
```

Then open <http://localhost:5000>.

## Choosing a model (`.env`)

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | tiny | Runs on almost any laptop |
| `Qwen/Qwen2.5-1.5B-Instruct` | small | **Default** — good balance |
| `meta-llama/Llama-3.2-1B-Instruct` | small | Needs HF access approval |
| `Qwen/Qwen2.5-3B-Instruct` | medium | Smarter; wants more RAM / a GPU |

## Project structure

| File | Purpose |
|------|---------|
| `agent.py` | Loads the local LLM + the reasoning / tool loop |
| `tools.py` | `web_search` and `read_url` web tools |
| `cli.py` | Command-line chat |
| `web_app.py` | Flask web chat server |
| `templates/index.html` | Web chat UI |
| `requirements.txt` | Python dependencies |
| `.env.example` | Optional model configuration |

## Notes

- Smaller local models are not as sharp as giant cloud models, but they are
  **fully yours, private, and free to run**. Pick a bigger model for more skill.
- Want zero Python setup? You can also run a local model with
  [Ollama](https://ollama.com) and point an OpenAI-compatible client at it —
  but this project keeps everything in pure Python so it truly is its own LLM.

---

Built for learning and tinkering. Have fun! 🚀

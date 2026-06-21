# SmartAI 🧠🔍📚

A **self-hosted** AI assistant that runs its **own local LLM** — no external API,
no API key, no per-message cost — and that you can **teach with your own data**.

SmartAI is built to be as smart as a local model can be:

1. 🤖 **Its own LLM** — an open-source instruct model that runs entirely on your
   machine (no OpenAI, no keys).
2. 🔍 **Live web search** — searches DuckDuckGo and reads pages for fresh facts.
3. 📚 **Your own knowledge base (RAG)** — feed it your notes/PDFs and it answers
   from *your* material, citing the file.
4. 🧩 **Self-critique** — it reviews and improves its own draft before replying.

> Honest note: a model that runs free on your laptop is far smaller than
> ChatGPT/GPT-4 and won't beat it in general. But by grounding answers in the
> **live web** and **your own documents**, SmartAI can be *more useful than
> ChatGPT on your specific topics* — because it has read things ChatGPT never has.

## Feed your AI your own data 📚

1. Put files in the `data/` folder (`.txt`, `.md`, `.csv`, or `.pdf`).
2. Run SmartAI — it indexes them on startup.
3. Ask away. It finds the most relevant passages and answers from them, naming
   the source file.

That's Retrieval-Augmented Generation (RAG): the model stays the same, but its
*knowledge* grows with whatever you give it.

## How it works

```
You ─► Local LLM (on your machine) ─► decides what to do next
                 ├► KB: query       ─► your own documents
                 ├► SEARCH: query   ─► live web results
                 ├► READ: url       ─► page text
                 └► ANSWER: ...      ─► then self-reviews → final answer
```

## Setup

```bash
git clone https://github.com/TigReh10/ai.git
cd ai
pip install -r requirements.txt
```

(Optional) pick a model in `.env` — the default works out of the box:

```bash
cp .env.example .env
```

The first run downloads the model weights automatically.

## Usage

```bash
python cli.py        # command-line chat
python web_app.py    # web UI at http://localhost:5000
```

## Choosing a model (`.env`)

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | tiny | Runs on almost any laptop |
| `Qwen/Qwen2.5-1.5B-Instruct` | small | **Default** — good balance |
| `Qwen/Qwen2.5-3B-Instruct` | medium | Smarter; wants more RAM / a GPU |
| `Qwen/Qwen2.5-7B-Instruct` | large | Much smarter; needs a strong GPU |

## Project structure

| File | Purpose |
|------|---------|
| `agent.py` | Local LLM + reasoning loop + self-critique |
| `knowledge.py` | Your personal knowledge base (RAG over `data/`) |
| `tools.py` | `web_search` and `read_url` web tools |
| `cli.py` | Command-line chat |
| `web_app.py` | Flask web chat server |
| `templates/index.html` | Web chat UI |
| `data/` | Drop your own files here to teach the AI |

---

Built for learning and tinkering. Have fun! 🚀

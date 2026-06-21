# SmartAI 🧠🔍

An actually smart, agentic AI assistant with **live web search** built in.

Unlike a plain chatbot, SmartAI reasons about your question and, when it needs
fresh or factual information, it **searches the web and reads pages on its own**
before answering — then cites its sources. It uses tool / function calling under
the hood, so the model decides *when* to look things up.

## Features

- 🔍 **Live web search** via DuckDuckGo (no paid search API needed)
- 📄 **Page reading** — fetches and extracts the readable text of any URL
- 🤖 **Agentic loop** — the model plans, searches, reads, then answers with sources
- 💬 **Two interfaces** — a command-line chat and a clean web UI
- 🔌 **Model-flexible** — works with OpenAI or any OpenAI-compatible endpoint
  (OpenRouter, Groq, Together, local Ollama / LM Studio, …)

## How it works

```
You ─► SmartAI (LLM) ─► decides: do I need the web?
                 │
                 ├► web_search(query)  ─► top results
                 ├► read_url(url)      ─► page text
                 └► writes a grounded answer with sources
```

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/TigReh10/ai.git
   cd ai
   pip install -r requirements.txt
   ```

2. **Add your API key**

   Copy the example env file and fill in your key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set `OPENAI_API_KEY`. You can get a key from
   <https://platform.openai.com/api-keys>. To use a free/local model instead,
   set `OPENAI_BASE_URL` and `AI_MODEL` to match your provider.

## Usage

### Command line

```bash
python cli.py
```

```
You: who won the latest F1 race and by how much?
  [tool] web_search({'query': 'latest F1 race result winner margin'})
  [tool] read_url({'url': 'https://...'})

SmartAI: ... (a fresh, sourced answer)
```

### Web app

```bash
python web_app.py
```

Then open <http://localhost:5000> and chat in your browser.

## Project structure

| File | Purpose |
|------|---------|
| `agent.py` | The SmartAI agent + tool-calling loop |
| `tools.py` | `web_search` and `read_url` web tools |
| `cli.py` | Command-line chat interface |
| `web_app.py` | Flask web chat server |
| `templates/index.html` | Web chat UI |
| `requirements.txt` | Python dependencies |
| `.env.example` | Configuration template |

## Notes

- Web search uses DuckDuckGo, which is free and needs no API key. Only the LLM
  requires a key.
- Keep your real `.env` private — it is already in `.gitignore`.

---

Built for learning and tinkering. Have fun! 🚀

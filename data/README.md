# Your data goes here 📚

This is SmartAI's **personal knowledge base**. Anything you put in this folder
becomes part of what your AI knows.

## How to feed your AI

1. Drop files into this `data/` folder. Supported types:
   - `.txt`, `.md`, `.markdown`, `.csv`, `.text`
   - `.pdf` (needs `pypdf`, included in requirements.txt)
2. Run the app (`python cli.py` or `python web_app.py`).
3. Ask questions about your material — SmartAI automatically finds the most
   relevant passages and uses them to answer, and cites which file they came from.

## Tips

- More, well-organized files = smarter answers on your topics.
- Use clear filenames (e.g. `business-studies-notes.md`) — they show up as the
  source of each answer.
- This `README.md` is ignored by the indexer, so it won't pollute results.

That's it — this is how SmartAI can outperform a generic assistant on *your*
subjects: it has read your actual notes. 🚀

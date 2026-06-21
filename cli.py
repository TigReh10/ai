"""Command-line chat with SmartAI.

Usage:
    python cli.py
"""
from __future__ import annotations

from dotenv import load_dotenv

from agent import SmartAI


def main() -> None:
    load_dotenv()
    ai = SmartAI()
    print("SmartAI is ready. Web search is ON. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        answer = ai.ask(user)
        print(f"\nSmartAI: {answer}\n")


if __name__ == "__main__":
    main()

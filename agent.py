"""SmartAI — a self-hosted AI agent that runs its OWN local LLM.

There is no external API and no API key. The language model (an open-source
instruct model) is downloaded from Hugging Face the first time you run it, then
runs entirely on your own machine. It can still search the live web on its own
using a simple ReAct-style protocol.
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import read_url, web_search

DEFAULT_MODEL = os.getenv("AI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

SYSTEM_PROMPT = """You are SmartAI, a careful and knowledgeable assistant that runs locally.
You can use the live web when a question needs fresh or factual information.

Use this exact protocol — output ONLY ONE action per reply:
- To search the web, reply with a single line:
  SEARCH: <your search query>
- To read a specific page, reply with a single line:
  READ: <url>
- When you have enough information, reply with:
  ANSWER: <your final answer, including any source URLs>

Think briefly, then act. If the question is simple and you already know the
answer, you may reply directly with ANSWER:.
"""


class SmartAI:
    """A conversational agent powered by a locally-run open-source LLM."""

    def __init__(self, model_name: str | None = None, max_new_tokens: int = 512) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        print(
            f"Loading local model: {self.model_name}\n"
            f"(the first run downloads the weights — this can take a while)\u2026"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Model loaded. SmartAI is running fully on your machine.\n")

    def _generate(self) -> str:
        """Run one generation step from the local model."""
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output[0][inputs.shape[-1]:], skip_special_tokens=True
        )
        return text.strip()

    def ask(self, user_message: str, max_steps: int = 5, verbose: bool = True) -> str:
        """Ask a question; the model reasons and uses web tools as needed."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_steps):
            reply = self._generate()
            self.messages.append({"role": "assistant", "content": reply})

            answer_match = re.search(r"ANSWER:\s*(.+)", reply, re.S)
            search_match = re.search(r"SEARCH:\s*(.+)", reply)
            read_match = re.search(r"READ:\s*(\S+)", reply)

            if answer_match:
                return answer_match.group(1).strip()

            if search_match:
                query = search_match.group(1).strip()
                if verbose:
                    print(f"  [web_search] {query}")
                try:
                    results = web_search(query)
                    obs = "\n".join(
                        f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results
                    ) or "No results found."
                except Exception as exc:  # keep the loop alive
                    obs = f"Search failed: {exc}"
                self.messages.append(
                    {"role": "user", "content": f"SEARCH RESULTS:\n{obs}"}
                )
                continue

            if read_match:
                url = read_match.group(1).strip()
                if verbose:
                    print(f"  [read_url] {url}")
                try:
                    page = read_url(url)
                except Exception as exc:
                    page = f"Failed to read page: {exc}"
                self.messages.append(
                    {"role": "user", "content": f"PAGE CONTENT ({url}):\n{page}"}
                )
                continue

            # No protocol keyword — treat the whole reply as the answer.
            return reply

        # Out of steps: force a final written answer.
        self.messages.append(
            {
                "role": "user",
                "content": "Please give your best final answer now, starting with ANSWER:.",
            }
        )
        final = self._generate()
        return re.sub(r"^.*?ANSWER:\s*", "", final, flags=re.S).strip() or final

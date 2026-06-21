"""SmartAI — a self-hosted AI agent that runs its OWN local LLM.

No external API and no API key. The language model (an open-source instruct
model) is downloaded from Hugging Face the first time you run it, then runs
entirely on your own machine.

To make it as smart as possible it can:
  1. Search the live web and read pages (fresh facts).
  2. Use YOUR own documents via a personal knowledge base (RAG).
  3. Critique and improve its own draft answer before replying.
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import read_url, web_search

DEFAULT_MODEL = os.getenv("AI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

SYSTEM_PROMPT = """You are SmartAI, a careful and knowledgeable assistant that runs locally.
You can use the live web AND the user's own knowledge base when helpful.

Use this exact protocol — output ONLY ONE action per reply:
- To search the web:            SEARCH: <your search query>
- To read a specific page:       READ: <url>
- To search the user's own docs: KB: <query>
- To give your final answer:     ANSWER: <final answer, with sources>

Think briefly, then act. Prefer the user's knowledge base for questions about
their own material, and the web for fresh or general facts. If you already know
the answer, you may reply directly with ANSWER:.
"""


class SmartAI:
    """A conversational agent powered by a locally-run open-source LLM."""

    def __init__(
        self,
        model_name: str | None = None,
        knowledge=None,
        reflect: bool = True,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.knowledge = knowledge
        self.reflect = reflect
        self.max_new_tokens = max_new_tokens
        print(
            f"Loading local model: {self.model_name}\n"
            "(the first run downloads the weights — this can take a while)..."
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

    def _inject_knowledge(self, user_message: str) -> None:
        """Automatically surface the most relevant personal notes (RAG)."""
        if not self.knowledge:
            return
        hits = self.knowledge.search(user_message, k=4)
        if not hits:
            return
        ctx = "\n\n".join(f"[from {h['source']}]\n{h['text']}" for h in hits)
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "RELEVANT PASSAGES FROM YOUR KNOWLEDGE BASE (use if helpful):\n"
                    + ctx
                ),
            }
        )

    def ask(self, user_message: str, max_steps: int = 5, verbose: bool = True) -> str:
        """Ask a question; the model reasons and uses tools / your data as needed."""
        self.messages.append({"role": "user", "content": user_message})
        self._inject_knowledge(user_message)

        draft = ""
        for _ in range(max_steps):
            reply = self._generate()
            self.messages.append({"role": "assistant", "content": reply})

            answer_match = re.search(r"ANSWER:\s*(.+)", reply, re.S)
            search_match = re.search(r"SEARCH:\s*(.+)", reply)
            read_match = re.search(r"READ:\s*(\S+)", reply)
            kb_match = re.search(r"KB:\s*(.+)", reply)

            if answer_match:
                draft = answer_match.group(1).strip()
                break

            if kb_match and self.knowledge:
                query = kb_match.group(1).strip()
                if verbose:
                    print(f"  [knowledge_base] {query}")
                hits = self.knowledge.search(query, k=4)
                obs = "\n\n".join(
                    f"[from {h['source']}]\n{h['text']}" for h in hits
                ) or "No matching notes found."
                self.messages.append(
                    {"role": "user", "content": f"KNOWLEDGE BASE RESULTS:\n{obs}"}
                )
                continue

            if search_match:
                query = search_match.group(1).strip()
                if verbose:
                    print(f"  [web_search] {query}")
                try:
                    results = web_search(query)
                    obs = "\n".join(
                        f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results
                    ) or "No results found."
                except Exception as exc:
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

            # No protocol keyword — treat the whole reply as a draft answer.
            draft = reply
            break

        if not draft:
            self.messages.append(
                {
                    "role": "user",
                    "content": "Give your best final answer now, starting with ANSWER:.",
                }
            )
            draft = re.sub(r"^.*?ANSWER:\s*", "", self._generate(), flags=re.S).strip()

        if self.reflect:
            draft = self._reflect(draft, verbose=verbose)
        return draft

    def _reflect(self, draft: str, verbose: bool = True) -> str:
        """One self-critique pass to catch errors and improve the answer."""
        if verbose:
            print("  [self-review] checking the draft for mistakes...")
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Critically review your draft answer below for any factual "
                    "errors, missing details, or unclear parts, using the "
                    "evidence already gathered. Then output the improved final "
                    "answer ONLY, starting with ANSWER:.\n\nDRAFT:\n" + draft
                ),
            }
        )
        improved = self._generate()
        self.messages.append({"role": "assistant", "content": improved})
        match = re.search(r"ANSWER:\s*(.+)", improved, re.S)
        return match.group(1).strip() if match else (improved.strip() or draft)

"""SmartAI — a small but genuinely capable AI agent with live web search.

The agent uses an OpenAI-compatible chat model and decides on its own when to
search the web or read a page, using tool / function calling. This is what makes
it feel "smart": it reasons about the question, gathers fresh evidence from the
web when needed, then answers with sources.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from tools import read_url, web_search

SYSTEM_PROMPT = """You are SmartAI, a helpful, accurate and thoughtful assistant.
You can search the live web and read web pages using the provided tools.

Rules:
- When a question depends on current events, facts, prices, news, schedules,
  releases, or anything that may have changed recently, ALWAYS use web_search
  first instead of guessing.
- For important factual questions, read one or two of the most relevant results
  with read_url before answering.
- Think step by step. Be concise, correct, and cite the source URLs you used.
- If the evidence is unclear or you are unsure, say so honestly.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for up-to-date information. Returns a list "
                "of results with title, url and snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 6).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Fetch a web page and return its readable text so you can "
                "extract details and quotes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to read."},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
    "read_url": read_url,
}


class SmartAI:
    """A conversational agent that can use web tools."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.getenv("AI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _run_tool(self, name: str, arguments: dict) -> str:
        fn = TOOL_IMPLEMENTATIONS.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        try:
            result = fn(**arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # keep the loop alive on tool errors
            return f"Tool '{name}' failed: {exc}"

    def ask(self, user_message: str, max_steps: int = 6, verbose: bool = True) -> str:
        """Ask a question and get an answer, using tools as needed."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
            msg = response.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if verbose:
                    print(f"  [tool] {call.function.name}({args})")
                output = self._run_tool(call.function.name, args)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": output,
                    }
                )

        # Final attempt without tools to force a written answer.
        final = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.3,
        )
        return final.choices[0].message.content or ""

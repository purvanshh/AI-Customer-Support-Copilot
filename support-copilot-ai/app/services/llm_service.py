from __future__ import annotations

import os
import textwrap
from typing import Literal, Optional

from app.core.config import get_settings


class LLMService:
    """
    LLM wrapper that supports:
      - mock generation (no external calls; good for local dev/tests)
      - OpenAI Chat Completions
      - Gemini (google-generativeai)

    The only required method for the RAG pipeline is `generate_response(prompt)`.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response string from a prompt.

        Args:
            prompt: Full instruction prompt including context.

        Returns:
            Response text.
        """
        backend = self.settings.llm_backend
        if backend == "mock":
            return self._generate_mock(prompt)
        if backend == "openai":
            return self._generate_openai(prompt)
        if backend == "gemini":
            return self._generate_gemini(prompt)
        raise ValueError(f"Unknown llm_backend: {backend}")

    def _generate_mock(self, prompt: str) -> str:
        """
        Offline deterministic response for tests.
        """
        # Simple heuristic: if we see a context block, reuse its first line.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        # Try to find the first context payload after [CONTEXT]
        context_start = None
        for i, ln in enumerate(lines):
            if ln.startswith("[CONTEXT"):
                context_start = i
                break
        if context_start is not None and context_start + 1 < len(lines):
            snippet = lines[context_start + 1]
            # Keep it short and readable.
            snippet = textwrap.shorten(snippet, width=180, placeholder="...")
            return (
                "Thanks for reaching out. Based on similar cases in the provided context, "
                f"the likely resolution starts with: {snippet}\n\n"
                "If this doesn't match your situation, please share any relevant details "
                "(account info, steps you've tried, and error messages) and I'll narrow it down."
            )
        return (
            "Thanks for reaching out. I don’t have enough context in the provided materials. "
            "Could you share what you’ve tried so far and any error messages?"
        )

    def _generate_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": self.settings.gemini_temperature,
            },
        )
        return (resp.text or "").strip()


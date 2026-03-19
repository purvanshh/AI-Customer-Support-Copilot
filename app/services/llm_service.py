from __future__ import annotations

import os
import re
import time
from typing import Literal

from app.core.config import get_settings


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(
        self,
        *,
        customer_query: str,
        context: str,
        tags: list[str] | None = None,
        llm_backend_override: str | None = None,
    ) -> tuple[str, dict]:
        backend = llm_backend_override or self.settings.llm_backend

        if backend == "mock":
            return self._generate_mock(customer_query=customer_query, context=context), {
                "model_used": "mock",
                "backend": "mock",
                "response_time_ms": None,
            }
        if backend == "openai":
            return self._generate_openai(customer_query=customer_query, context=context, tags=tags), {
                "model_used": self.settings.openai_model,
                "backend": "openai",
                "response_time_ms": None,
            }
        if backend == "gemini":
            return self._generate_gemini(customer_query=customer_query, context=context, tags=tags), {
                "model_used": self.settings.gemini_model,
                "backend": "gemini",
                "response_time_ms": None,
            }

        raise ValueError(f"Unknown llm backend: {backend}")

    def _generate_mock(self, *, customer_query: str, context: str) -> str:
        # Pick the first "Historical Response:" chunk if present.
        m = re.search(r"Historical Response:\s*(.+)", context, flags=re.IGNORECASE | re.DOTALL)
        snippet = m.group(1).strip() if m else ""
        snippet = snippet.split("\n")[0].strip() if snippet else ""
        if not snippet:
            snippet = "we’ve seen similar issues before and the fix usually involves verifying your setup and trying a couple quick steps."

        tag_hint = ""
        if "billing" in context.lower() or "refund" in context.lower():
            tag_hint = "For billing-related cases, ensure the payment method and plan status are correct."

        return (
            "Thanks for reaching out. \n\n"
            f"Based on similar past tickets: {snippet}\n\n"
            "If you can share any relevant details (for example, the account/email involved and what you tried so far), "
            "I can help confirm the best next step."
            + (f"\n\n{tag_hint}" if tag_hint else "")
        )

    def _generate_openai(self, *, customer_query: str, context: str, tags: list[str] | None) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are SupportCopilot AI, a customer support assistant. "
            "Use the provided context to draft a helpful, concise, friendly reply. "
            "If context is insufficient, ask 1-2 clarifying questions instead of guessing."
        )

        tags_line = f"Ticket tags: {', '.join(tags)}" if tags else ""
        user_prompt = (
            f"{tags_line}\n\n"
            f"New customer ticket:\n{customer_query}\n\n"
            f"Relevant context from historical tickets and docs:\n{context}\n\n"
            "Write the response message body only. No JSON, no markdown code fences."
        )

        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
            max_tokens=self.settings.openai_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        # keep time for caller
        _ = time.time() - t0
        return text.strip()

    def _generate_gemini(self, *, customer_query: str, context: str, tags: list[str] | None) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)

        tags_line = f"Ticket tags: {', '.join(tags)}" if tags else ""
        prompt = (
            f"{tags_line}\n\n"
            f"New customer ticket:\n{customer_query}\n\n"
            f"Relevant context from historical tickets and docs:\n{context}\n\n"
            "Write the response message body only. No JSON."
        )

        resp = model.generate_content(
            prompt,
            generation_config={"temperature": self.settings.gemini_temperature, "max_output_tokens": self.settings.openai_max_tokens},
        )
        return (resp.text or "").strip()


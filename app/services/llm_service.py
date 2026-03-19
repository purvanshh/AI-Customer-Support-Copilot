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
            return self._generate_mock(customer_query=customer_query, context=context, tags=tags), {
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

    def _generate_mock(self, *, customer_query: str, context: str, tags: list[str] | None = None) -> str:
        """
        Improved mock generator: extracts the best-matching historical response
        from context and composes a structured reply. This replaces the old
        approach of grabbing only the *first* "Historical Response:" line — which
        was often from the wrong ticket when retrieval order was random.
        """

        # ── 1. Extract ALL historical responses from the context ──
        # Each ticket chunk looks like:
        #   Customer Query: ...
        #   Historical Response: ...
        responses: list[str] = []
        for m in re.finditer(r"Historical Response:\s*(.+?)(?=\nTicket ID:|\n\[|\Z)", context, flags=re.DOTALL):
            snippet = m.group(1).strip().split("\n")[0].strip()
            if snippet:
                responses.append(snippet)

        # ── 2. Also check for feedback-corrected responses ──
        for m in re.finditer(r"Corrected Response:\s*(.+?)(?=\n\[|\Z)", context, flags=re.DOTALL):
            snippet = m.group(1).strip().split("\n")[0].strip()
            if snippet:
                # Prioritize corrected responses (agent-verified quality)
                responses.insert(0, snippet)

        # ── 3. Also check for doc excerpts ──
        doc_snippets: list[str] = []
        for m in re.finditer(r"Doc Excerpt[^:]*:\s*(.+?)(?=\n\[|\Z)", context, flags=re.DOTALL):
            snippet = m.group(1).strip()
            # Take first meaningful non-header line
            for line in snippet.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and len(line) > 20:
                    doc_snippets.append(line[:300])
                    break

        # ── 4. Pick the best response ──
        best = ""
        if responses:
            best = responses[0]
        elif doc_snippets:
            best = doc_snippets[0]

        if not best:
            best = (
                "We've seen similar issues before. Could you share a bit more detail "
                "about your setup so we can pinpoint the issue?"
            )

        # ── 5. Build tag-aware context hints ──
        tag_hint = ""
        if tags:
            tag_lower = {t.lower() for t in tags}
            if "billing" in tag_lower:
                tag_hint = "\n\nFor billing-related cases, please verify your payment method and plan status are correct."
            elif "refund" in tag_lower:
                tag_hint = "\n\nRegarding refunds, our standard processing time is 3-5 business days."
            elif "technical issue" in tag_lower:
                tag_hint = "\n\nIf the issue persists after trying the steps above, please share your browser/OS version and any error messages."

        return (
            f"Thanks for reaching out.\n\n"
            f"Based on similar past cases: {best}\n\n"
            f"If you can share any relevant details (account email, steps tried so far), "
            f"I can help confirm the best next step."
            f"{tag_hint}"
        )

    def _generate_openai(self, *, customer_query: str, context: str, tags: list[str] | None) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are SupportCopilot AI, a customer support assistant. "
            "ONLY use the provided context to draft your reply. "
            "If the context does not contain enough information, say: "
            "'I don't have enough information to answer this fully — could you provide more details?' "
            "Do NOT hallucinate or make up information. "
            "Be helpful, concise, and friendly."
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
            "ONLY use the context above. If context is insufficient, ask clarifying questions. "
            "Write the response message body only. No JSON."
        )

        resp = model.generate_content(
            prompt,
            generation_config={"temperature": self.settings.gemini_temperature, "max_output_tokens": self.settings.openai_max_tokens},
        )
        return (resp.text or "").strip()

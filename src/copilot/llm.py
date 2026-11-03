"""LLM provider wrapper (Groq / Gemini) with local fallback."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from .config import GEMINI_API_KEY, GROQ_API_KEY, GROQ_MODEL

_CACHE: dict[str, str] = {}

GROUNDED_PROMPT = """You are an onboarding assistant. Answer the employee's question using ONLY the
past tickets below. If they do not contain the answer, reply exactly: NO_ANSWER.
Cite sources as [ticket_id]. Be concise and practical.

PAST TICKETS:
{context}

QUESTION: {question}
"""


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt(question: str, context: str) -> str:
    return GROUNDED_PROMPT.format(context=context, question=question)


def _call_groq(prompt: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise, factual onboarding knowledge assistant."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=600,
        )
        text = (chat_completion.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        print(f"[llm] Groq call failed: {e}")
        return None


def _call_gemini(prompt: str) -> str | None:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text or None
    except Exception as e:
        print(f"[llm] Gemini call failed: {e}")
        return None


def generate(question: str, context: str, fallback_answer: str | None = None) -> str:
    """
    Grounded generation. Checks Groq first if GROQ_API_KEY is configured,
    then Gemini if GEMINI_API_KEY is configured,
    or falls back to the top retrieved ticket answer verbatim.
    """
    prompt = build_prompt(question, context)
    key = _cache_key(prompt)
    if key in _CACHE:
        return _CACHE[key]

    text = None
    if GROQ_API_KEY:
        text = _call_groq(prompt)

    if text is None and GEMINI_API_KEY:
        text = _call_gemini(prompt)

    if text is None:
        # Local fallback: return top ticket answer verbatim if available
        if fallback_answer and fallback_answer.strip():
            text = fallback_answer.strip()
        else:
            text = "NO_ANSWER"

    _CACHE[key] = text
    return text


def is_no_answer(text: str) -> bool:
    return text.strip().upper().startswith("NO_ANSWER")

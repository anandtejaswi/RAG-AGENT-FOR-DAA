"""Model configuration.

The chat model is selected from environment variables so the same pipeline runs
against an OpenAI-compatible endpoint (OpenRouter) or Google Gemini without a
code change. Values are read from .env at the project root.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "LLM_PROVIDER": "openai_compat",
    "LLM_MODEL": "z-ai/glm-5.3-flash",
    "LLM_BASE_URL": "https://openrouter.ai/api/v1",
}


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    # LangSmith / LangChain Tracing configuration
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if tracing_enabled or api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "rag-aktu"))
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))


# Ensure environment is loaded on module import
load_env()


def setting(name: str) -> str | None:
    # Map clean name with legacy fallback if needed
    val = os.environ.get(name)
    if not val and name.startswith("LLM_"):
        legacy = "MUNSHI_" + name[4:]
        val = os.environ.get(legacy)
    return val or DEFAULTS.get(name)


@lru_cache(maxsize=8)
def chat_model(temperature: float = 0.0, max_tokens: int = 4096,
               reasoning_effort: str | None = None):
    """Build the LangChain chat model named by the environment.

    `reasoning_effort` matters on reasoning models served through OpenRouter:
    max_tokens covers reasoning *and* content, so a model that reasons freely can
    spend the whole budget and return empty content. Structured-output callers
    (the query router, the grading judge) pass "low" to keep the budget for the
    JSON they need. The endpoint for GLM refuses to disable reasoning outright.
    """
    provider = setting("LLM_PROVIDER")
    model = setting("LLM_MODEL")
    api_key = setting("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key or setting("GOOGLE_API_KEY"),
        )

    if provider == "openai_compat":
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set; add it to .env")
        extra: dict = {}
        if reasoning_effort:
            extra["reasoning"] = {"effort": reasoning_effort}
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=setting("LLM_BASE_URL"),
            timeout=180,
            max_retries=3,
            extra_body=extra or None,
        )

    raise ValueError(f"unsupported LLM_PROVIDER: {provider!r}")


def model_label() -> str:
    return f"{setting('LLM_MODEL')} via {setting('LLM_PROVIDER')}"

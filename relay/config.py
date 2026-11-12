"""Shared settings. The Groq key is read from .env and never printed or logged."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_BASE = "https://api.groq.com/openai/v1"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def groq_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        raise SystemExit("GROQ_API_KEY missing from .env")
    return k


# Headers worth keeping from every response (never the request's Authorization).
RL_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    "x-groq-region", "x-request-id", "date",
)

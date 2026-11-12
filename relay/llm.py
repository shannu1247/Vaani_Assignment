"""One streaming chat client for both the real Groq API and the stand-in.

Both speak the OpenAI-compatible /chat/completions SSE protocol, so the
pipeline cannot tell them apart except by base URL. Every call returns a
CallResult with client-side timings (ttft, total), token usage, Groq's own
server-side split (queue / prompt / completion time) when present, and the
rate-limit headers. No automatic retries here: retry policy lives in the
pipeline so every retry is visible and counted.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from .config import RL_HEADERS


def prompt_key(model: str, messages: list[dict]) -> str:
    """Key for replaying a recorded real reply in the stand-in. The model is
    deliberately NOT part of the key: when the pipeline fails over to another
    model under load, the stand-in still replays real text for that prompt
    (known gap: on the stand-in a fallback model is not actually weaker)."""
    import hashlib
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:24]


def redact(s: str) -> str:
    """Groq 429 bodies name the organization id; never let it reach a log."""
    return re.sub(r"org_[A-Za-z0-9]+", "org_REDACTED", s)


@dataclass
class CallResult:
    status: str                    # ok | rate_limited | timeout | error
    http_status: int | None = None
    text: str = ""
    reasoning: str = ""
    ttft_ms: float | None = None   # first content OR reasoning token
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    server: dict = field(default_factory=dict)   # x_groq usage timings (s)
    headers: dict = field(default_factory=dict)
    error: str = ""
    retry_after_s: float | None = None
    finish_reason: str | None = None


class LLMClient:
    def __init__(self, base_url: str, api_key: str, *, connect_timeout: float = 5.0,
                 max_concurrency: int | None = None):
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(max_connections=max_concurrency or 200,
                              max_keepalive_connections=max_concurrency or 50)
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(None, connect=connect_timeout), limits=limits)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, model: str, messages: list[dict], *, max_tokens: int = 512,
                   temperature: float = 0.0, ttft_timeout_s: float = 10.0,
                   total_timeout_s: float = 30.0, extra: dict | None = None,
                   on_token=None) -> CallResult:
        """Stream one completion. ttft_timeout_s bounds the wait for the first
        token; total_timeout_s bounds the whole call (slow-but-succeeding calls
        are cut off rather than allowed to hold a worker)."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if extra:
            body.update(extra)
        t0 = time.perf_counter()
        res = CallResult(status="error")
        deadline = t0 + total_timeout_s
        try:
            async with self._client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body,
                    timeout=httpx.Timeout(total_timeout_s, connect=5.0,
                                          read=ttft_timeout_s)) as r:
                res.http_status = r.status_code
                res.headers = {k: r.headers[k] for k in RL_HEADERS if k in r.headers}
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", "replace")
                    res.error = redact(raw)[:500]
                    res.total_ms = (time.perf_counter() - t0) * 1000
                    if r.status_code == 429:
                        res.status = "rate_limited"
                        ra = r.headers.get("retry-after")
                        res.retry_after_s = float(ra) if ra else None
                    elif r.status_code in (408, 504):
                        res.status = "timeout"
                    return res
                async for line in r.aiter_lines():
                    if time.perf_counter() > deadline:
                        raise httpx.ReadTimeout("total deadline exceeded")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if "error" in ev:
                        res.error = json.dumps(ev["error"])[:500]
                        res.status = "error"
                        res.total_ms = (time.perf_counter() - t0) * 1000
                        return res
                    for ch in ev.get("choices", []):
                        d = ch.get("delta", {})
                        piece = d.get("content") or ""
                        rpiece = d.get("reasoning") or ""
                        if (piece or rpiece) and res.ttft_ms is None:
                            res.ttft_ms = (time.perf_counter() - t0) * 1000
                        res.text += piece
                        res.reasoning += rpiece
                        if piece and on_token:
                            on_token(piece)
                        if ch.get("finish_reason"):
                            res.finish_reason = ch["finish_reason"]
                    usage = ev.get("usage") or (ev.get("x_groq") or {}).get("usage")
                    if usage:
                        res.tokens_in = usage.get("prompt_tokens", 0)
                        res.tokens_out = usage.get("completion_tokens", 0)
                        res.server = {k: usage[k] for k in
                                      ("queue_time", "prompt_time", "completion_time",
                                       "total_time") if k in usage}
                res.status = "ok"
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            res.status, res.error = "timeout", f"{type(e).__name__}: {e}"[:300]
        except httpx.HTTPError as e:
            res.status, res.error = "error", f"{type(e).__name__}: {e}"[:300]
        res.total_ms = (time.perf_counter() - t0) * 1000
        return res

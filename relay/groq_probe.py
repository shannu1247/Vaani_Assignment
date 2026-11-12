"""Measure the real Groq service. Every request is logged raw (timings,
usage, server-side split, rate-limit headers, error body) to
results/probe/<stage>.jsonl. Stages are run one at a time so the daily
budget spent on probing is known in advance.

  headers  : 1 tiny call per chat model -> limits the server reports     (~3 req)
  latency  : N sequential realistic calls (RAG-sized prompt), one model  (N req)
  rpm      : fire tiny calls concurrently past the RPM limit, observe 429,
             then poll once per second until it recovers                  (~RPM+20 req)
  tpm      : big-prompt calls back-to-back until the token limit trips    (~3-6 req)
  inflight : K simultaneous calls, check whether concurrency is capped    (K req)

Run:  .venv\\Scripts\\python -m relay.groq_probe <stage> [--model M] [--n N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

from .config import GROQ_BASE, RESULTS, groq_key
from .llm import CallResult, LLMClient

OUT = RESULTS / "probe"
OUT.mkdir(parents=True, exist_ok=True)
CHAT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def extra_for(model: str) -> dict:
    # Reasoning models: keep reasoning short so latency reflects a production setting.
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def log(stage: str, model: str, r: CallResult, **kw) -> dict:
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "stage": stage, "model": model, "status": r.status,
           "http_status": r.http_status, "ttft_ms": r.ttft_ms,
           "total_ms": round(r.total_ms, 1), "tokens_in": r.tokens_in,
           "tokens_out": r.tokens_out, "server": r.server, "headers": r.headers,
           "retry_after_s": r.retry_after_s, "error": r.error,
           "finish_reason": r.finish_reason, "reasoning_chars": len(r.reasoning),
           "text_head": r.text[:120], **kw}
    with (OUT / f"{stage}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def tiny() -> list[dict]:
    return [{"role": "user", "content": "Reply with the single word: ok"}]


def rag_prompt(i: int) -> list[dict]:
    """A realistic pipeline-sized prompt (~1.2-1.5k tokens) built from real
    paper text so latency reflects what the drafting step will send."""
    from pathlib import Path
    blocks = [json.loads(l) for l in (Path(__file__).resolve().parent.parent /
              "papers" / "blocks.jsonl").open(encoding="utf-8")]
    body = [b for b in blocks if b["kind"] == "body" and len(b["text"]) > 300]
    start = (i * 7) % (len(body) - 8)
    ctx = "\n\n".join(f"[{b['short']} p.{b['page']} §{b['sec_num']}] {b['text']}"
                      for b in body[start:start + 8])
    q = ["What is the main contribution described?", "What numbers are reported?",
         "Which component is introduced here and why?"][i % 3]
    return [{"role": "system", "content": "Answer only from the passages and cite them as [paper p.N §S]."},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"}]


async def stage_headers(c: LLMClient, args) -> None:
    for m in CHAT_MODELS:
        r = await c.chat(m, tiny(), max_tokens=64, extra=extra_for(m))
        row = log("headers", m, r)
        print(m, row["status"], row["http_status"], json.dumps(row["headers"]),
              row["server"], row["tokens_in"], row["tokens_out"])


async def stage_latency(c: LLMClient, args) -> None:
    for i in range(args.n):
        r = await c.chat(args.model, rag_prompt(i), max_tokens=args.max_tokens,
                         extra=extra_for(args.model), ttft_timeout_s=30, total_timeout_s=60)
        row = log("latency", args.model, r, i=i)
        print(i, row["status"], row["ttft_ms"] and round(row["ttft_ms"]), row["total_ms"],
              row["tokens_in"], row["tokens_out"], row["server"],
              row["headers"].get("x-ratelimit-remaining-tokens"))
        if r.status == "rate_limited":
            await asyncio.sleep((r.retry_after_s or 5) + 0.5)
        else:
            await asyncio.sleep(args.gap)


async def stage_rpm(c: LLMClient, args) -> None:
    """Burst past RPM with tiny requests, then poll 1/s until accepted again."""
    t0 = time.perf_counter()

    async def one(k):
        r = await c.chat(args.model, tiny(), max_tokens=8, extra=extra_for(args.model))
        return log("rpm", args.model, r, k=k, phase="burst",
                   t_rel_ms=round((time.perf_counter() - t0) * 1000))

    rows = await asyncio.gather(*(one(k) for k in range(args.n)))
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"burst: sent={len(rows)} ok={ok} 429={sum(r['status']=='rate_limited' for r in rows)}")
    first429 = next((r for r in rows if r["status"] == "rate_limited"), None)
    if first429:
        print("429 body:", first429["error"])
        print("429 headers:", json.dumps(first429["headers"]))
    t429 = time.perf_counter()
    for k in range(120):
        r = await c.chat(args.model, tiny(), max_tokens=8, extra=extra_for(args.model))
        row = log("rpm", args.model, r, k=k, phase="recover",
                  since_429_ms=round((time.perf_counter() - t429) * 1000))
        print(f"  recover poll {k}: {row['status']} retry_after={row['retry_after_s']} "
              f"since={row['since_429_ms']}ms")
        if r.status == "ok":
            break
        await asyncio.sleep(1.0)


async def stage_tpm(c: LLMClient, args) -> None:
    """Send large prompts back to back until the per-minute token limit trips."""
    big = rag_prompt(0)
    big[1]["content"] = big[1]["content"] * args.mult
    t0 = time.perf_counter()
    for k in range(args.n):
        r = await c.chat(args.model, big, max_tokens=16, extra=extra_for(args.model),
                         ttft_timeout_s=30, total_timeout_s=60)
        row = log("tpm", args.model, r, k=k, t_rel_ms=round((time.perf_counter() - t0) * 1000))
        print(k, row["status"], row["tokens_in"], row["headers"].get("x-ratelimit-remaining-tokens"),
              row["retry_after_s"], row["error"][:300])
        if r.status == "rate_limited":
            await asyncio.sleep((r.retry_after_s or 5) + 0.2)
            r2 = await c.chat(args.model, tiny(), max_tokens=8, extra=extra_for(args.model))
            log("tpm", args.model, r2, k=k, phase="after_retry_after")
            print("  after retry-after (tiny):", r2.status)
            break


async def stage_inflight(c: LLMClient, args) -> None:
    t0 = time.perf_counter()

    async def one(k):
        r = await c.chat(args.model, rag_prompt(k), max_tokens=200,
                         extra=extra_for(args.model), ttft_timeout_s=30, total_timeout_s=60)
        return log("inflight", args.model, r, k=k,
                   t_rel_ms=round((time.perf_counter() - t0) * 1000))

    rows = await asyncio.gather(*(one(k) for k in range(args.n)))
    for r in sorted(rows, key=lambda r: r["k"]):
        print(r["k"], r["status"], r["ttft_ms"] and round(r["ttft_ms"]), r["total_ms"],
              r["server"].get("queue_time"), r["error"][:120])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["headers", "latency", "rpm", "tpm", "inflight"])
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--mult", type=int, default=3)
    args = ap.parse_args()
    c = LLMClient(GROQ_BASE, groq_key())
    try:
        await globals()[f"stage_{args.stage}"](c, args)
    finally:
        await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())

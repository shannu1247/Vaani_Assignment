"""Stand-in model service: an OpenAI/Groq-compatible /chat/completions
endpoint whose behaviour is calibrated from results/probe/*.jsonl.

What it reproduces (all measured on the real service, see calibrate()):
  * latency = network RTT + queue + prompt processing + streamed completion,
    each drawn from the measured per-model distributions;
  * queue time grows with the number of requests in flight (load slowdown);
  * limits are continuously refilling buckets per model, like Groq:
      RPM 30, TPM 8000 charged as prompt_tokens + max_tokens at admission,
      RPD 1000 refilling one request every 86.4 s;
    a refused call gets HTTP 429 with Groq's error JSON and retry-after;
  * failures: 5xx errors, a mid-stream disconnect, and "slow success" stalls;
  * replies are streamed a few tokens per SSE event, with usage + x_groq
    timings in the final event.

Reply content: a real Groq reply recorded for the same prompt is replayed
when one exists (results/replay.jsonl, written by the pipeline on real runs);
otherwise an extractive answer is synthesised from the passages in the prompt.

Runtime knobs: POST /admin/config {...} (e.g. {"rpm": 15} to halve the
per-minute limit, {"rpd_remaining": 0} to exhaust the day), GET /admin/stats.

Run:  .venv\\Scripts\\python -m relay.standin --port 8088
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "results" / "probe"
REPLAY = ROOT / "results" / "replay.jsonl"


# ---------------------------------------------------------------- calibration
@dataclass
class ModelProfile:
    rtt_ms: tuple[float, float]          # lognormal (mu, sigma) of network + TLS overhead
    queue_ms: tuple[float, float]        # lognormal of server queue time at low load
    prompt_tok_per_s: float
    out_tok_per_s: float
    reasoning_tokens: tuple[int, int]    # hidden reasoning tokens before content (gpt-oss)


def _lognorm(x: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(x, 1e-3, None))
    return float(x.mean()), float(max(x.std(), 0.05))


def calibrate() -> dict[str, ModelProfile]:
    rows = [json.loads(l) for l in (PROBE / "latency.jsonl").open(encoding="utf-8")]
    out = {}
    for m in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == m and r["status"] == "ok" and r["server"]]
        srv = np.array([r["server"]["total_time"] * 1000 for r in R])
        q = np.array([r["server"]["queue_time"] * 1000 for r in R])
        tot = np.array([r["total_ms"] for r in R])
        pin = np.array([r["tokens_in"] for r in R])
        pt = np.array([r["server"]["prompt_time"] for r in R])
        tout = np.array([r["tokens_out"] for r in R])
        ct = np.array([r["server"]["completion_time"] for r in R])
        rc = np.array([r.get("reasoning_chars", 0) for r in R])
        out[m] = ModelProfile(
            rtt_ms=_lognorm(tot - srv - q),
            queue_ms=_lognorm(q),
            prompt_tok_per_s=float(np.median(pin / pt)),
            out_tok_per_s=float(np.median(tout / ct)),
            reasoning_tokens=(int(np.percentile(rc / 4, 25)), int(np.percentile(rc / 4, 75)) + 1))
    return out


# ---------------------------------------------------------------- limits
class Bucket:
    """Continuously refilling bucket (what Groq's reset headers imply)."""

    def __init__(self, capacity: float, per_seconds: float):
        self.capacity, self.per = capacity, per_seconds
        self.level, self.t = capacity, time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.t) * self.capacity / self.per)
        self.t = now

    def wait_for(self, amount: float) -> float:
        self._refill()
        return 0.0 if self.level >= amount else (amount - self.level) * self.per / self.capacity

    def take(self, amount: float):
        self._refill()
        self.level -= amount

    def reset_s(self) -> float:
        self._refill()
        return (self.capacity - self.level) * self.per / self.capacity


@dataclass
class Config:
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 1000
    tpd: int = 200_000                # measured: TPD Limit 200000, ~2.315 tok/s refill
    error_rate: float = 0.01          # 5xx before streaming
    drop_rate: float = 0.005          # disconnect mid-stream
    slow_rate: float = 0.03           # "succeeds slowly": long stall before first token
    slow_stall_s: tuple[float, float] = (4.0, 12.0)
    load_knee: int = 8                # in-flight requests before queueing slows down
    load_ms_per_req: float = 120.0    # extra queue per in-flight request beyond the knee
    time_scale: float = 1.0           # <1 speeds the whole service up for quick tests
    rpd_remaining: float | None = None  # set to exhaust the day for all models
    seed: int = 7


@dataclass
class State:
    cfg: Config
    profiles: dict[str, ModelProfile]
    buckets: dict[str, dict[str, Bucket]] = field(default_factory=dict)
    in_flight: int = 0
    stats: dict = field(default_factory=lambda: {"ok": 0, "429": 0, "5xx": 0, "drop": 0,
                                                  "slow": 0, "peak_in_flight": 0})

    def model_buckets(self, m: str) -> dict[str, Bucket]:
        if m not in self.buckets:
            self.buckets[m] = {"rpm": Bucket(self.cfg.rpm, 60), "tpm": Bucket(self.cfg.tpm, 60),
                               "rpd": Bucket(self.cfg.rpd, 86400),
                               "tpd": Bucket(self.cfg.tpd, 86400)}
            if self.cfg.rpd_remaining is not None:      # day already spent for every model
                self.buckets[m]["rpd"].level = float(self.cfg.rpd_remaining)
                self.buckets[m]["tpd"].level = 0.0
        return self.buckets[m]


def fmt_dur(s: float) -> str:
    if s >= 60:
        return f"{int(s // 60)}m{s % 60:.3f}s"
    return f"{s * 1000:.0f}ms" if s < 1 else f"{s:.3f}s"


def est_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages) // 4 + 8 * len(messages)


def load_replay() -> dict[str, dict]:
    d = {}
    if REPLAY.exists():
        for l in REPLAY.open(encoding="utf-8"):
            r = json.loads(l)
            d[r["key"]] = r
    return d


from .llm import prompt_key  # noqa: E402  (shared with the recorder in pipeline)


def synth_reply(messages: list[dict]) -> str:
    """Fallback content when no real reply was recorded for this prompt."""
    sys = messages[0].get("content", "") if messages else ""
    user = messages[-1].get("content", "")
    if "TASK: understand" in sys:
        q = user.split("Question:", 1)[-1].strip()
        return json.dumps({"in_scope": True, "papers": [], "queries": [q], "kind": "fact"})
    if "TASK: judge" in sys:
        return json.dumps({"verdict": "supported"})
    # draft: copy the most question-like passage sentence with its id
    q = set(re.findall(r"[a-z]{4,}", user.split("Question:", 1)[-1].lower()))
    best, best_s = None, 0
    for cid, body in re.findall(r"\[([A-Z]\d+)\][^\n]*\n(.+?)(?=\n\[[A-Z]\d+\]|\nQuestion:|$)", user, re.S):
        for s in re.split(r"(?<=[.!?])\s+", body):
            sc = len(q & set(re.findall(r"[a-z]{4,}", s.lower())))
            if sc > best_s and 30 < len(s) < 400:
                best, best_s = (cid, s.strip()), sc
    if not best or best_s < 2:
        return "NOT_IN_PAPERS"
    return f"{best[1]} [{best[0]}]"


# ---------------------------------------------------------------- app
def make_app(cfg: Config) -> FastAPI:
    app = FastAPI()
    st = State(cfg=cfg, profiles=calibrate())
    replay = load_replay()
    rng = random.Random(cfg.seed)

    def profile(m: str) -> ModelProfile:
        return st.profiles.get(m) or next(iter(st.profiles.values()))

    def rl_headers(m: str) -> dict:
        b = st.model_buckets(m)
        return {"x-ratelimit-limit-requests": str(cfg.rpd),
                "x-ratelimit-limit-tokens": str(cfg.tpm),
                "x-ratelimit-remaining-requests": str(int(b["rpd"].level)),
                "x-ratelimit-remaining-tokens": str(int(b["tpm"].level)),
                "x-ratelimit-reset-requests": fmt_dur(b["rpd"].reset_s()),
                "x-ratelimit-reset-tokens": fmt_dur(b["tpm"].reset_s()),
                "x-groq-region": "standin", "x-request-id": f"req_{uuid.uuid4().hex[:24]}"}

    def too_many(m: str, kind: str, limit: int, used: int, req: int, wait: float):
        names = {"rpm": "requests per minute (RPM)", "tpm": "tokens per minute (TPM)",
                 "rpd": "requests per day (RPD)", "tpd": "tokens per day (TPD)"}
        st.stats["429"] += 1
        h = rl_headers(m)
        h["retry-after"] = str(max(1, math.ceil(wait)))
        msg = (f"Rate limit reached for model `{m}` in organization `org_STANDIN` service tier "
               f"`on_demand` on {names[kind]}: Limit {limit}, Used {used}, Requested {req}. "
               f"Please try again in {fmt_dur(wait)}.")
        return JSONResponse(status_code=429, headers=h, content={"error": {
            "message": msg, "type": "tokens" if kind in ("tpm", "tpd") else "requests",
            "code": "rate_limit_exceeded"}})

    @app.post("/admin/config")
    async def set_config(req: Request):
        body = await req.json()
        for k, v in body.items():
            if k == "rpd_remaining":
                for b in st.buckets.values():
                    b["rpd"].level = float(v)
                    b["tpd"].level = 0.0
                st.cfg.rpd_remaining = float(v)
            elif hasattr(st.cfg, k):
                setattr(st.cfg, k, type(getattr(st.cfg, k))(v) if not isinstance(v, list) else tuple(v))
        if any(k in body for k in ("rpm", "tpm", "rpd")):
            for b in st.buckets.values():   # resize, keep current fill fraction
                for name, cap in (("rpm", st.cfg.rpm), ("tpm", st.cfg.tpm), ("rpd", st.cfg.rpd)):
                    frac = b[name].level / b[name].capacity
                    b[name].capacity, b[name].level = cap, cap * frac
        return {"config": st.cfg.__dict__}

    @app.post("/admin/reset")
    async def reset():
        st.buckets.clear()
        st.cfg.rpd_remaining = None
        st.stats.update({k: 0 for k in st.stats})
        return {"ok": True}

    @app.get("/admin/stats")
    async def stats():
        return {"stats": st.stats, "in_flight": st.in_flight,
                "profiles": {m: p.__dict__ for m, p in st.profiles.items()}}

    @app.post("/openai/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        m = body.get("model", "")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 1024)
        p_in = est_tokens(messages)
        p = profile(m)
        b = st.model_buckets(m)
        ts = cfg.time_scale

        # --- admission: same order Groq reports (day, minute requests, minute tokens)
        for kind, amount, limit in (("rpd", 1, cfg.rpd), ("rpm", 1, cfg.rpm),
                                    ("tpm", p_in + max_tokens, cfg.tpm),
                                    ("tpd", p_in + max_tokens, cfg.tpd)):
            w = b[kind].wait_for(amount)
            if w > 0:
                return too_many(m, kind, limit, int(limit - b[kind].level), amount, w)
        for kind, amount in (("rpd", 1), ("rpm", 1), ("tpm", p_in + max_tokens)):
            b[kind].take(amount)
        # unused completion budget is NOT refunded within the minute (matches what
        # remaining-tokens showed on the real service right after a call)

        if rng.random() < cfg.error_rate:
            st.stats["5xx"] += 1
            await asyncio.sleep(rng.uniform(0.2, 1.0) * ts)
            return JSONResponse(status_code=503, content={"error": {
                "message": "Service Unavailable", "type": "internal_server_error"}})

        # --- reply content
        fam = "draft" if "TASK: draft" in (messages[0].get("content", "") if messages else "") else "other"
        rec = replay.get(prompt_key(m, messages))
        text = rec["text"] if rec else synth_reply(messages)
        out_toks = max(1, len(text) // 4)
        if out_toks > max_tokens:
            text, out_toks = text[: max_tokens * 4], max_tokens
        reasoning = rng.randint(*p.reasoning_tokens) if p.reasoning_tokens[1] > 1 else 0
        b["tpd"].take(p_in + out_toks + reasoning)       # actual use, like Groq

        st.in_flight += 1
        st.stats["peak_in_flight"] = max(st.stats["peak_in_flight"], st.in_flight)
        over = max(0, st.in_flight - cfg.load_knee)
        rtt = rng.lognormvariate(*p.rtt_ms)
        queue = rng.lognormvariate(*p.queue_ms) + over * cfg.load_ms_per_req * rng.uniform(0.5, 1.5)
        prompt_s = p_in / p.prompt_tok_per_s
        slow = rng.random() < cfg.slow_rate
        stall = rng.uniform(*cfg.slow_stall_s) if slow else 0.0
        if slow:
            st.stats["slow"] += 1
        drop_at = rng.randint(1, max(1, out_toks)) if rng.random() < cfg.drop_rate else None
        per_tok = 1.0 / p.out_tok_per_s * (1 + over * 0.03)   # decode slows a little under load
        rid = f"chatcmpl-{uuid.uuid4().hex[:20]}"

        async def stream():
            try:
                # measured: the ~390 ms client-side overhead is all before the
                # first token (ttft - queue - prompt_time), none after the last
                await asyncio.sleep((rtt + queue + stall * 1000) / 1000 * ts
                                    + (prompt_s + reasoning * per_tok) * ts)
                pieces = re.findall(r"\S+\s*|\s+", text)
                sent, i = 0, 0
                while i < len(pieces):
                    n = rng.randint(1, 4)
                    chunk = "".join(pieces[i:i + n])
                    i += n
                    sent += max(1, len(chunk) // 4)
                    if drop_at is not None and sent >= drop_at:
                        st.stats["drop"] += 1
                        yield "data: " + json.dumps({"error": {"message": "connection reset",
                                                               "type": "server_error"}}) + "\n\n"
                        return
                    ev = {"id": rid, "object": "chat.completion.chunk", "model": m,
                          "choices": [{"index": 0, "delta": {"content": chunk},
                                       "finish_reason": None}]}
                    yield "data: " + json.dumps(ev) + "\n\n"
                    await asyncio.sleep(max(1, len(chunk) // 4) * per_tok * ts)
                usage = {"prompt_tokens": p_in, "completion_tokens": out_toks + reasoning,
                         "total_tokens": p_in + out_toks + reasoning,
                         "queue_time": (queue + stall * 1000) / 1000, "prompt_time": prompt_s,
                         "completion_time": (out_toks + reasoning) * per_tok,
                         "total_time": prompt_s + (out_toks + reasoning) * per_tok}
                fin = {"id": rid, "object": "chat.completion.chunk", "model": m,
                       "choices": [{"index": 0, "delta": {}, "finish_reason":
                                    "length" if out_toks >= max_tokens else "stop"}],
                       "usage": usage, "x_groq": {"id": rid, "usage": usage}}
                yield "data: " + json.dumps(fin) + "\n\n"
                yield "data: [DONE]\n\n"
                st.stats["ok"] += 1
            finally:
                st.in_flight -= 1

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers=rl_headers(m))

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--time-scale", type=float, default=1.0)
    args = ap.parse_args()
    import uvicorn
    cfg = Config(time_scale=args.time_scale)
    for m, p in calibrate().items():
        print(f"calibrated {m}: {p}")
    uvicorn.run(make_app(cfg), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

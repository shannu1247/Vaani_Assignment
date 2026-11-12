"""The Relay pipeline.

  0. cache / coalesce   identical question already answered or in flight -> share it
  1. understand  (LLM)  scope check, target paper(s), search rewrites   [skippable]
  2. retrieve    (tool) hybrid search (+ table/figure lookup tool)
  3. decide     (branch) refuse early / pick a model the budget can serve now / shed
  4. draft       (LLM)  streamed, cited answer
  5. verify      (code) per sentence, while streaming; only verified sentences are sent
  6. loop               NOT_IN_PAPERS or nothing verifiable -> one wider retrieval + redraft

Every request keeps a timeline of (step, start_ms, end_ms, kind) with kind in
{wait, work, retry} so a slow request can be split into waiting / working /
retrying after the fact (WRITEUP section 4).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .governor import ModelGovernor
from .llm import CallResult, LLMClient, prompt_key
from .retrieval import Chunk, Setting, approx_tokens, get_index
from .verify import CITE, CITE_ID, REFUSAL, StreamVerifier

PAPER_NAMES = {"Transformer": "Attention Is All You Need (arXiv 1706.03762v7)",
               "Moshi": "Moshi (arXiv 2410.00037v1)",
               "H-Net": "Dynamic Chunking / H-Net (arXiv 2507.07955v2)"}

UNDERSTAND_SYS = """TASK: understand
You route questions for an assistant that answers ONLY from three papers:
"Transformer" = Attention Is All You Need; "Moshi" = Moshi speech-text dialogue model (includes the Mimi codec and the Helium text LM); "H-Net" = Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.
Reply with JSON only:
{"in_scope": true|false, "papers": [subset of "Transformer","Moshi","H-Net"], "queries": [1-3 short search queries in the papers' own terminology], "kind": "fact"|"compare"|"explain"}
in_scope is false only for requests unrelated to ML research (greetings, weather, coding help, etc.). Questions about these papers that they might not answer are still in_scope."""

DRAFT_SYS = """TASK: draft
You answer engineers' questions using ONLY the numbered passages below, taken from three research papers.
Rules:
1. End every sentence that states a fact with the id(s) of the passage(s) that support it, like [M73] or [T12, T13].
2. Use only what the passages say. Never use outside knowledge, even if you are sure of it.
3. Copy numbers exactly as written in the passages, and attribute them to the paper they come from.
4. If the passages do not contain the answer, reply with exactly: NOT_IN_PAPERS
5. If they answer only part of the question, answer that part and add one sentence saying what the papers do not state.
6. At most 120 words. Plain sentences: no headings, no tables, no bullet lists."""


def prompt_tokens(messages: list[dict]) -> int:
    """Estimated prompt tokens for reservation. chars/4 undercounts the
    paper text (measured on 63 real draft calls: real/est p50 1.05, p95 1.20,
    max 1.29), so reserve at the p95 ratio; headers correct the rest."""
    return int(1.2 * sum(approx_tokens(m["content"]) + 8 for m in messages))


def normalize(q: str) -> str:
    return re.sub(r"[^a-z0-9.\- ]+", "", q.lower()).strip(" ?.")


def heuristic_understand(q: str) -> dict:
    ql = q.lower()
    papers = []
    if re.search(r"moshi|mimi|helium|inner monologue|speech|audio|codec", ql):
        papers.append("Moshi")
    if re.search(r"h-net|hnet|dynamic chunk|routing module|byte-level|tokenizer-free|chunk", ql):
        papers.append("H-Net")
    if re.search(r"transformer|attention is all|bleu|wmt|encoder-decoder|multi-head", ql) \
            and "Moshi" not in papers:
        papers.append("Transformer")
    return {"in_scope": True, "papers": papers, "queries": [q], "kind": "fact", "source": "heuristic"}


@dataclass
class Policy:
    name: str = "smart"
    draft_chain: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    understand_model: str = "qwen/qwen3.8-27b"
    use_understand: bool = True
    # "ambiguous": call the LLM only when the keyword heuristic finds no paper
    # or several. Measured on the 26 eval questions (c320_k5): LLM rewrites
    # gave hit@5 20/20 vs 19/20 raw, no end-to-end difference, but cost
    # ~0.8 s TTFT (474 ms call + ~350 ms extra retrieval) and the most
    # expensive call per answer (qwen $0.80/$4.00 per M, a Preview model).
    understand_mode: str = "ambiguous"     # "always" | "ambiguous" | "never"
    ttft_budget_s: float = 6.0        # must be able to start drafting within this
    call_ttft_timeout_s: float = 4.0  # measured p99 ~2.1 s -> 4 s cuts "slow success" stalls
    call_total_timeout_s: float = 20.0
    max_retries: int = 2
    quality_wait_s: float = 1.0       # longest wait accepted to keep the stronger model
    loop_max: int = 1
    cache: bool = True
    coalesce: bool = True
    naive_retry_429: bool = False     # naive: sleep(backoff) and resend to the same model
    use_governor: bool = True         # naive: no local limit mirror, learn only from 429s
    refuse_rerank_below: float = -4.0 # cross-encoder logit; only used when rerank is on
    draft_max_tokens: dict = field(default_factory=lambda: {
        "openai/gpt-oss-120b": 520, "openai/gpt-oss-20b": 520, "qwen/qwen3.8-27b": 300})

    def extra(self, model: str) -> dict:
        if model.startswith("openai/gpt-oss"):
            return {"reasoning_effort": "low"}
        if model.startswith("qwen/"):
            return {"reasoning_effort": "none"}
        return {}


@dataclass
class Record:
    request_id: str
    question: str
    scenario: str
    source: str
    retrieval_setting: str
    start_time: str
    t0: float = 0.0
    first_token_ms: float | None = None
    total_ms: float = 0.0
    status: str = "ok"            # ok | rate_limited | timeout | error
    outcome: str = ""             # answered | refused | busy | daily_exhausted | gave_up | ...
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    degraded: list[str] = field(default_factory=list)
    cache: str = ""               # hit | coalesced | ""
    answer: str = ""
    raw_draft: str = ""
    rejected: list = field(default_factory=list)
    cited: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    understanding: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def span(self, step: str, start: float, kind: str, **kw) -> None:
        self.timeline.append({"step": step, "start_ms": start, "end_ms": self.ms(),
                              "kind": kind, **kw})


class Relay:
    def __init__(self, base_url: str, api_key: str, *, source: str, setting: Setting,
                 policy: Policy | None = None, replay_path=None, limits: dict | None = None,
                 retrieval_workers: int = 4):
        self.client = LLMClient(base_url, api_key)
        self.source = source
        self.setting = setting
        self.policy = policy or Policy()
        lim = limits or {}
        models = set(self.policy.draft_chain) | {self.policy.understand_model}
        self.govs = {m: ModelGovernor(m, **lim) for m in models}
        self.answers: dict[str, dict] = {}
        self.live: dict[str, "Record"] = {}
        self.inflight: dict[str, asyncio.Future] = {}
        self.pool = ThreadPoolExecutor(retrieval_workers)
        self.replay_path = replay_path
        self.index = get_index(setting.chunk_tokens)
        # warm up: load embedder (+ re-ranker) now, not on the first user's request
        # (smoke test: the first retrieval took 5.2 s cold vs ~0.45 s warm)
        self.index.search(["warm up"], setting)

    async def aclose(self) -> None:
        await self.client.aclose()
        self.pool.shutdown(wait=False)

    def peak_queue(self) -> int:
        return max(g.peak_queue for g in self.govs.values())

    # ---------------------------------------------------------------- model call
    async def _call(self, rec: Record, step: str, model: str, messages: list[dict],
                    max_tokens: int, deadline: float, priority: int,
                    on_token: Callable[[str], None] | None = None,
                    json_mode: bool = False) -> CallResult | None:
        """Admission through the governor, then one streamed call. None = the
        governor could not grant capacity before `deadline`."""
        gov = self.govs[model]
        need = prompt_tokens(messages) + max_tokens
        t = rec.ms()
        granted = None
        try:
            granted = await gov.acquire(need, deadline, priority) if self.policy.use_governor else True
        finally:   # also when the user gives up mid-wait, so section-4 splits stay complete
            rec.span(f"{step}:queue", t, "wait", model=model, granted=granted, tokens=need)
        if not granted:
            return None
        extra = dict(self.policy.extra(model))
        if json_mode and self.source == "real":
            extra["response_format"] = {"type": "json_object"}
        t = rec.ms()
        r = await self.client.chat(model, messages, max_tokens=max_tokens, extra=extra,
                                   ttft_timeout_s=self.policy.call_ttft_timeout_s,
                                   total_timeout_s=self.policy.call_total_timeout_s,
                                   on_token=on_token)
        gov.observe(r.headers, r.status, r.retry_after_s, r.error)
        if self.policy.use_governor and r.status != "ok" and r.http_status in (None, 429):
            # The call never consumed provider capacity (connection failed, or
            # the provider refused it). Without this refund a network blip left
            # phantom reservations and later requests queued 13-22 s for nothing
            # (seen in results/runs_network_outage/).
            gov.refund(need, request=True)
        rec.tokens_in += r.tokens_in
        rec.tokens_out += r.tokens_out
        gov.spend_day(r.tokens_in + r.tokens_out)
        rec.calls.append({"step": step, "model": model, "status": r.status,
                          "http": r.http_status, "ttft_ms": r.ttft_ms,
                          "total_ms": round(r.total_ms, 1), "server": r.server,
                          "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                          "retry_after": r.retry_after_s, "error": r.error[:200],
                          "text": r.text[:1500]})
        rec.span(f"{step}:call", t, "work" if r.status == "ok" else "retry", model=model,
                 status=r.status, server_queue_ms=round(r.server.get("queue_time", 0) * 1000, 1))
        if r.status == "ok" and self.source == "real" and self.replay_path:
            with open(self.replay_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": prompt_key(model, messages), "model": model,
                                    "text": r.text}, ensure_ascii=False) + "\n")
        return r

    # ---------------------------------------------------------------- steps
    async def understand(self, rec: Record, q: str) -> dict:
        p = self.policy
        h = heuristic_understand(q)
        if not p.use_understand or p.understand_mode == "never":
            return h
        if p.understand_mode == "ambiguous" and len(h["papers"]) == 1:
            return h
        gov = self.govs[p.understand_model]
        if gov.estimate_wait(400) > 0.5:          # never queue for an optional step
            rec.degraded.append("no_understand")
            return heuristic_understand(q)
        msgs = [{"role": "system", "content": UNDERSTAND_SYS},
                {"role": "user", "content": f"Question: {q}"}]
        r = await self._call(rec, "understand", p.understand_model, msgs, 150,
                             time.monotonic() + 0.5, priority=6, json_mode=True)
        if r is None or r.status != "ok":
            rec.degraded.append("no_understand")
            return heuristic_understand(q)
        try:
            j = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))
            j["papers"] = [x for x in j.get("papers", []) if x in PAPER_NAMES]
            j["queries"] = [s for s in j.get("queries", []) if isinstance(s, str)][:3]
            j["source"] = "llm"
            return j
        except Exception:
            rec.degraded.append("understand_unparsed")
            return heuristic_understand(q)

    def _lookup_tool(self, q: str, papers: list[str]) -> list[Chunk]:
        """Second tool: exact table/figure/section lookup. A question naming
        'Table 3' is ambiguous across papers (each has one); only resolve it
        when the paper is known."""
        out = []
        for kind, n in re.findall(r"\b(Table|Figure|Fig\.)\s*(\d+)", q):
            pat = re.compile(rf"^{kind}\s+{n}[:.]")
            for c in self.index.chunks:
                if (not papers or c.short in papers) and any(pat.match(l) for l in c.text.splitlines()):
                    out.append(c)
        return out[:2] if papers else []

    async def retrieve(self, rec: Record, q: str, u: dict, setting: Setting) -> list[tuple[Chunk, float]]:
        queries = [q] + [s for s in u.get("queries", []) if s.lower() != q.lower()]
        t = rec.ms()
        loop = asyncio.get_running_loop()
        try:
            hits, info = await asyncio.wait_for(
                loop.run_in_executor(self.pool, self.index.search, queries, setting,
                                     u.get("papers") or None), timeout=8.0)
        except asyncio.TimeoutError:
            rec.span("retrieve", t, "retry", note="timeout")
            rec.degraded.append("retrieve_timeout")
            fallback = Setting(setting.chunk_tokens, setting.k, False)
            hits, info = await loop.run_in_executor(self.pool, self.index.search, [q], fallback, None)
        extra = [c for c in self._lookup_tool(q, u.get("papers", [])) if c.cid not in {h.cid for h, _ in hits}]
        hits = hits + [(c, 0.0) for c in extra]
        rec.span("retrieve", t, "work", **info, lookup=len(extra))
        rec.retrieved = [(c.cid, round(s, 3)) for c, s in hits]
        return hits

    def draft_messages(self, q: str, hits: list[tuple[Chunk, float]], feedback: str = "") -> list[dict]:
        ctx = "\n\n".join(f"[{c.cid}] ({c.header}, p.{c.page_from})\n{c.text}" for c, _ in hits)
        return [{"role": "system", "content": DRAFT_SYS},
                {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"
                                            + (f"\n\n{feedback}" if feedback else "")}]

    def render(self, sent: str, by_id: dict[str, Chunk]) -> str:
        def rep(m):
            ids = CITE_ID.findall(m.group(0))
            return " (" + "; ".join(dict.fromkeys(by_id[i].cite for i in ids if i in by_id)) + ")"
        return CITE.sub(rep, sent)

    async def draft(self, rec: Record, q: str, hits, emit, deadline_start: float,
                    feedback: str = "") -> str:
        """Returns 'answered' | 'not_in_papers' | 'unverifiable' | 'busy' | 'failed'."""
        p = self.policy
        by_id = {c.cid: c for c, _ in hits}
        passages = {c.cid: f"{c.short} {c.sec_title} {c.text}" for c, _ in hits}
        msgs = self.draft_messages(q, hits, feedback)
        attempts = 0
        tried: list[str] = []
        chain = list(p.draft_chain)
        while attempts <= p.max_retries:
            # pick the best model whose budget can start us before the deadline
            model = None
            if not p.use_governor:
                model = chain[0]
            # Best model that can start within quality_wait_s; otherwise whichever
            # can start soonest before the deadline (real run: a user gave up
            # after 4.5 s while we waited for gpt-oss-120b instead of starting on 20b).
            soonest, soonest_w = None, float("inf")
            for m in ([] if model else chain):
                if m in tried and len(tried) < len(chain):
                    continue
                need = prompt_tokens(msgs) + p.draft_max_tokens[m]
                w = self.govs[m].estimate_wait(need)
                if time.monotonic() + w > deadline_start:
                    continue
                if w <= p.quality_wait_s:
                    model = m
                    break
                if w < soonest_w:
                    soonest, soonest_w = m, w
            model = model or soonest
            if model is None:
                if all(self.govs[m].day_remaining() < 1 for m in chain):
                    return "daily_exhausted"
                return "busy"
            if model != chain[0]:
                rec.degraded.append(f"model:{model.split('/')[-1]}")
            ver = StreamVerifier(passages)
            sent_any = False

            def on_token(piece: str):
                nonlocal sent_any
                for s in ver.feed(piece):
                    sent_any = True
                    emit(self.render(s, by_id) + " ")

            r = await self._call(rec, "draft", model, msgs, p.draft_max_tokens[model],
                                 deadline_start, priority=3 if attempts == 0 else 2,
                                 on_token=on_token)
            if r is None:                 # lost the race for capacity: try the next model
                tried.append(model)
                continue
            rec.model = model
            if r.status == "ok":
                for s in ver.finish():
                    sent_any = True
                    emit(self.render(s, by_id) + " ")
                rec.raw_draft = r.text
                rec.rejected = ver.rejected
                rec.cited = sorted(ver.cited)
                if "NOT_IN_PAPERS" in r.text and not ver.released:
                    return "not_in_papers"
                if not ver.cited:
                    return "unverifiable"
                return "answered"
            # failed call
            if sent_any:                  # never resend after the user saw text
                emit(" [The answer was cut short by a provider error.]")
                rec.rejected = ver.rejected
                return "answered"
            attempts += 1
            rec.retries += 1
            if r.status == "rate_limited" and p.naive_retry_429:
                t = rec.ms()
                await asyncio.sleep(min(2 ** attempts, 8))
                rec.span("draft:backoff", t, "retry")
                continue                   # same model again
            tried.append(model)            # smart: move to the next model's budget
        return "failed"

    # ---------------------------------------------------------------- request
    async def answer(self, question: str, *, scenario: str, emit: Callable[[str], None],
                     request_id: str | None = None) -> Record:
        rec = Record(request_id=request_id or uuid.uuid4().hex[:12], question=question,
                     scenario=scenario, source=self.source,
                     retrieval_setting=self.setting.name,
                     start_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                     t0=time.perf_counter())
        self.live[rec.request_id] = rec      # so a caller that cancels us can still log what we spent
        out: list[str] = []

        def send(s: str):
            if rec.first_token_ms is None and s.strip():
                rec.first_token_ms = rec.ms()
            out.append(s)
            emit(s)

        key = normalize(question)
        p = self.policy
        try:
            if p.cache and key in self.answers:
                rec.cache = "hit"
                cached = self.answers[key]
                send(cached["answer"])
                rec.outcome = cached["outcome"]
                rec.cited = cached.get("cited", [])
                return rec
            if p.coalesce and key in self.inflight:
                rec.cache = "coalesced"
                t = rec.ms()
                cached = await asyncio.shield(self.inflight[key])
                rec.span("coalesce_wait", t, "wait")
                if cached:
                    send(cached["answer"])
                    rec.outcome = cached["outcome"]
                    rec.status = cached["status"]
                    rec.cited = cached.get("cited", [])
                    return rec
            fut = asyncio.get_running_loop().create_future()
            self.inflight[key] = fut
            try:
                await self._run(rec, question, send)
            finally:
                result = {"answer": "".join(out), "outcome": rec.outcome, "status": rec.status,
                          "cited": list(rec.cited)}
                if not fut.done():
                    fut.set_result(result if rec.outcome in ("answered", "refused") else None)
                self.inflight.pop(key, None)
                if p.cache and rec.outcome in ("answered", "refused") \
                        and not any(d.startswith("model:") for d in rec.degraded):
                    self.answers[key] = result
        except asyncio.CancelledError:
            rec.status, rec.outcome = "timeout", "gave_up"
            raise
        finally:
            rec.answer = "".join(out)
            rec.total_ms = rec.ms()
        return rec

    async def _run(self, rec: Record, q: str, send) -> None:
        p = self.policy
        deadline_start = time.monotonic() + p.ttft_budget_s
        u = await self.understand(rec, q)
        rec.understanding = u
        if not u.get("in_scope", True):
            send("I can only answer questions about three papers: "
                 + "; ".join(PAPER_NAMES.values()) + ". This question is outside them.")
            rec.outcome = "refused"
            return
        setting = self.setting
        feedback = ""
        for it in range(p.loop_max + 1):
            hits = await self.retrieve(rec, q, u, setting)
            if setting.rerank and hits and hits[0][1] < p.refuse_rerank_below:
                rec.span("decide", rec.ms(), "work", branch="refuse_low_score")
                verdict = "not_in_papers"
            else:
                verdict = await self.draft(rec, q, hits, send, deadline_start, feedback)
            # Loop only when widening can help: the target paper was missed by
            # retrieval, or the draft cited nothing we could verify. A refusal
            # after looking in the right paper is accepted (a redraft would
            # double the token cost of every unanswerable question).
            target = set(u.get("papers") or [])
            missed_target = bool(target) and not (target & {c.short for c, _ in hits[:3]})
            if (verdict == "unverifiable" or (verdict == "not_in_papers" and missed_target)) \
                    and it < p.loop_max and time.monotonic() < deadline_start - 1.0:
                # loop: widen the search once (more pieces, no paper preference)
                rec.span("decide", rec.ms(), "work", branch=f"loop_after_{verdict}")
                setting = Setting(setting.chunk_tokens, setting.k * 2, setting.rerank, setting.pool)
                u = {**u, "papers": []}
                if verdict == "unverifiable" and rec.rejected:
                    # a redraft must differ from the first draft: at temperature 0
                    # the same prompt repeats the same mistake (seen on real Groq)
                    feedback = ("Your previous answer could not be checked: "
                                + "; ".join(r for _, r in rec.rejected[:3])
                                + ". Cite only the bracketed passage ids shown above, exactly as written.")
                continue
            break
        by_cite = {c.cid: c for c, _ in hits}
        if verdict == "answered":
            rec.outcome = "answered"
        elif verdict in ("not_in_papers", "unverifiable"):
            secs = []
            for c, _ in hits[:3]:
                if c.cite not in secs:
                    secs.append(c.cite)
            send("The three papers do not say this. I looked in: " + "; ".join(secs) + "."
                 if verdict == "not_in_papers" else
                 "I could not find a statement in the papers that answers this, so I will not guess. "
                 "Closest passages: " + "; ".join(secs) + ".")
            rec.outcome = "refused"
        elif verdict == "daily_exhausted":
            rec.status, rec.outcome = "rate_limited", "daily_exhausted"
            refill = min(self.govs[m].day_refill_s() for m in p.draft_chain)
            send(daily_exhausted_reply(hits, refill))
        elif verdict == "busy":
            rec.status, rec.outcome = "rate_limited", "busy"
            send(busy_reply(hits))
        else:
            rec.status, rec.outcome = "error", "failed"
            send("Sorry, the answer service failed twice in a row. Please try again in a minute. "
                 "Relevant passages: " + "; ".join(c.cite for c, _ in hits[:3]) + ".")
        _ = by_cite


def _passage_list(hits) -> str:
    seen, out = set(), []
    for c, _ in hits[:3]:
        if c.cite in seen:
            continue
        seen.add(c.cite)
        snippet = re.sub(r"\s+", " ", c.text)[:160].rsplit(" ", 1)[0]
        out.append(f"- {c.cite}: \"{snippet}…\"")
    return "\n".join(out)


def daily_exhausted_reply(hits, refill_s: float = 0.0) -> str:
    mins = max(1, round(refill_s / 60))
    return ("Today's budget for written answers is used up, so I can't write and check a new "
            "answer right now, and I won't guess. Search still works, and these passages look "
            "most relevant to your question:\n"
            + _passage_list(hits) +
            f"\nQuestions already answered today are still answered instantly. The budget refills "
            f"gradually rather than at midnight: the next new answer can be written in about "
            f"{mins} minute{'s' if mins != 1 else ''}, and full capacity returns over 24 hours.")


def busy_reply(hits) -> str:
    return ("We're getting more questions than we can answer carefully right now, so I didn't "
            "start one I couldn't finish in time. The most relevant passages:\n" + _passage_list(hits) +
            "\nPlease ask again in about a minute.")

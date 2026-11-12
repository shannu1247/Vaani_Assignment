"""Run the pipeline: quality evaluation or load scenarios, real or stand-in.

  eval : every question in eval/questions.yaml, one at a time, paced so no
         request waits on the rate limit (TTFT then measures the pipeline,
         not queueing we caused ourselves).
  load : a scenario from SCENARIOS: an arrival schedule of simulated users,
         each with a patience; a user who sees no first token within their
         patience gives up and the request is cancelled (it stops spending budget).

Writes results/results.csv (the brief's 12 columns, appended) and the full
per-request record (timeline, calls, answer, rejected sentences) to
results/runs/<scenario>.jsonl.

  python -m relay.run eval --source real --setting 320,5,0
  python -m relay.run load --source stand-in --scenario burst
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml

from .config import GROQ_BASE, RESULTS, groq_key
from .grade import grade
from .pipeline import Policy, Record, Relay
from .retrieval import Setting

ROOT = Path(__file__).resolve().parent.parent
CSV = RESULTS / "results.csv"
RUNS = RESULTS / "runs"
RUNS.mkdir(exist_ok=True)
COLS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
        "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
STANDIN = "http://127.0.0.1:8088/openai/v1"


def load_questions() -> list[dict]:
    return yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))


def write(rec: Record, relay: Relay, run_tag: str) -> dict:
    by_id = {c.cid: c for c in relay.index.chunks}
    cited = [(by_id[i].short, by_id[i].page_from, by_id[i].page_to) for i in rec.cited if i in by_id]
    g = grade(rec.question, rec.outcome, rec.answer, cited)
    row = {"request_id": rec.request_id, "scenario": rec.scenario, "source": rec.source,
           "start_time": rec.start_time,
           "first_token_ms": "" if rec.first_token_ms is None else round(rec.first_token_ms, 1),
           "total_ms": round(rec.total_ms, 1), "status": rec.status, "retries": rec.retries,
           "tokens_in": rec.tokens_in, "tokens_out": rec.tokens_out,
           "retrieval_setting": rec.retrieval_setting, "answer_grade": g}
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    full = asdict(rec)
    full.pop("t0", None)
    full.update({"answer_grade": g, "run": run_tag})
    with (RUNS / f"{rec.scenario}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(full, ensure_ascii=False, default=str) + "\n")
    return row


def make_relay(source: str, setting: Setting, policy: Policy) -> Relay:
    if source == "real":
        return Relay(GROQ_BASE, groq_key(), source="real", setting=setting, policy=policy,
                     replay_path=RESULTS / "replay.jsonl")
    return Relay(STANDIN, "standin", source="stand-in", setting=setting, policy=policy)


# ------------------------------------------------------------------ eval
async def run_eval(args) -> None:
    c, k, rr = (int(x) for x in args.setting.split(","))
    setting = Setting(c, k, bool(rr))
    model = args.model
    policy = Policy(draft_chain=(model,), ttft_budget_s=120.0, cache=False, coalesce=False,
                    understand_mode=args.understand)
    relay = make_relay(args.source, setting, policy)
    scenario = args.scenario or f"eval_{setting.name}"
    qs = load_questions()
    if args.only:
        qs = [q for q in qs if q["id"] in args.only.split(",")]
    gov = relay.govs[model]
    # resume: skip questions this scenario already answered cleanly (multi-day runs)
    done_q = set()
    prev = RUNS / f"{scenario}.jsonl"
    if prev.exists():
        done_q = {json.loads(l)["question"] for l in prev.open(encoding="utf-8")}
    try:
        for q in qs:
            if q["q"] in done_q:
                continue
            # pace: start only when the draft can be admitted immediately
            need = 2600
            while gov.estimate_wait(need) > 0 or relay.govs[policy.understand_model].estimate_wait(500) > 0:
                await asyncio.sleep(max(0.2, min(gov.estimate_wait(need), 10)))
            rec = await relay.answer(q["q"], scenario=scenario, emit=lambda s: None)
            # An eval row must measure the pipeline, not an outage or a spent day:
            # such requests are logged as evidence and the run stops.
            per_day = [c for c in rec.calls if c["status"] == "rate_limited"
                       and re.search(r"per day|TPD|RPD", c["error"])]
            net = [c for c in rec.calls if c["http"] is None and c["status"] != "ok"]
            if per_day or net:
                ev = asdict(rec)
                ev.pop("t0", None)
                with (RESULTS / "probe" / "daily_limit_and_outages.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                why = per_day[0]["error"] if per_day else net[0]["error"]
                print(f"STOP at {q['id']}: {'daily limit' if per_day else 'network'}: {why[:300]}")
                break
            row = write(rec, relay, args.tag)
            print(f"{q['id']} {row['answer_grade']:<17} ttft={row['first_token_ms']!s:>7} "
                  f"total={row['total_ms']:>7} in={rec.tokens_in:>5} out={rec.tokens_out:>4} "
                  f"{rec.outcome:<9} {('rej=' + str(len(rec.rejected))) if rec.rejected else ''} "
                  f"{','.join(rec.degraded)} :: {rec.answer[:150]!r}")
    finally:
        await relay.aclose()


# ------------------------------------------------------------------ load
def schedule(name: str, rng: random.Random, qs: list[dict]) -> tuple[list[tuple[float, str]], dict]:
    """Arrival times (s from start) and questions. Returns (arrivals, standin_config)."""
    hot = [q["q"] for q in qs if q["id"] in ("q04", "q01", "u01")]
    pool = [q["q"] for q in qs]

    def poisson(rate_per_min: float, start: float, end: float, hot_frac: float = 0.0):
        t, out = start, []
        while True:
            t += rng.expovariate(rate_per_min / 60)
            if t >= end:
                return out
            q = rng.choice(hot) if rng.random() < hot_frac else rng.choice(pool)
            out.append((t, q))

    if name == "steady":                       # comfortably below capacity
        return poisson(6, 0, 300), {}
    if name == "burst":                        # incident: 45 users in ~20 s, many ask the same thing
        return poisson(3, 0, 240) + [(60 + rng.uniform(0, 20),
                                      rng.choice(hot) if rng.random() < 0.5 else rng.choice(pool))
                                     for _ in range(45)], {}
    if name == "overload":                     # sustained ~2.5x capacity
        return poisson(24, 0, 300), {}
    if name == "rpm_halved":                   # provider halves per-minute limits without notice
        return poisson(6, 0, 300), {"rpm": 15, "tpm": 4000}
    if name == "daily_exhausted":              # the day's request budget is gone
        return poisson(6, 0, 180), {"rpd_remaining": 0}
    if name == "naive_retry_overload":         # same overload, naive retry policy (section 6)
        return poisson(24, 0, 300), {}
    if name == "real_steady":
        return poisson(4, 0, 300), {}
    if name == "real_primary_day_exhausted":  # real: gpt-oss-120b's TPD already spent
        return poisson(4, 0, 240), {}
    if name == "real_burst":
        return [(rng.uniform(0, 10), rng.choice(hot) if rng.random() < 0.5 else rng.choice(pool))
                for _ in range(16)], {}
    raise SystemExit(f"unknown scenario {name}")


async def run_load(args) -> None:
    c, k, rr = (int(x) for x in args.setting.split(","))
    setting = Setting(c, k, bool(rr))
    rng = random.Random(args.seed)
    qs = load_questions()
    arrivals, cfg = schedule(args.scenario, rng, qs)
    arrivals.sort()
    naive = args.scenario.startswith("naive")
    policy = Policy(naive_retry_429=naive, max_retries=4 if naive else 2)
    if naive:   # a typical first version: one model, blind sends, backoff-and-retry, no cache
        policy.cache = policy.coalesce = policy.use_governor = False
        policy.draft_chain = (policy.draft_chain[0],)
        policy.ttft_budget_s = 60.0
    if args.source == "stand-in":
        async with httpx.AsyncClient() as h:
            await h.post(STANDIN.replace("/openai/v1", "/admin/reset"))
            base = {"rpm": 30, "tpm": 8000, "rpd": 1000}
            await h.post(STANDIN.replace("/openai/v1", "/admin/config"), json={**base, **cfg})
    # The client is NOT told about halved limits or an exhausted day: its
    # governor starts from the measured defaults and must learn the change
    # from response headers and 429 bodies, as it would in production.
    relay = make_relay(args.source, setting, policy)
    print(f"{args.scenario}: {len(arrivals)} requests over {arrivals[-1][0]:.0f}s "
          f"source={args.source} setting={setting.name} cfg={cfg}")
    t_start = time.monotonic()
    rows: list[dict] = []

    async def user(i: int, at: float, q: str):
        await asyncio.sleep(max(0.0, t_start + at - time.monotonic()))
        patience = min(20.0, max(3.0, rng.lognormvariate(math.log(8.0), 0.4)))
        first = asyncio.Event()

        def emit(s: str):
            if s.strip():
                first.set()

        rid = f"{args.scenario[:6]}-{args.source[:4]}-{i:04d}"
        task = asyncio.create_task(relay.answer(q, scenario=args.scenario, emit=emit,
                                                request_id=rid))
        waiter = asyncio.create_task(first.wait())
        done, _ = await asyncio.wait({task, waiter}, timeout=patience,
                                     return_when=asyncio.FIRST_COMPLETED)
        if not done:                      # user gave up before seeing anything
            if naive:
                # a naive server does not notice the user left: the work runs on and
                # its tokens are spent anyway (counted in this row once it finishes)
                try:
                    await asyncio.wait_for(task, timeout=90)
                except (asyncio.TimeoutError, Exception):
                    pass
            else:
                task.cancel()             # stop spending budget on someone who left
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            rec = relay.live.pop(rid)     # keeps start_time, tokens already spent, timeline
            if rec.first_token_ms is not None and rec.first_token_ms > patience * 1000:
                rec.first_token_ms = None  # produced after the user left: never seen
            rec.status, rec.outcome = "timeout", "gave_up"
            rec.total_ms = round(patience * 1000, 1)
            rows.append(write(rec, relay, f"{args.tag};patience={patience:.1f}"))
            return
        waiter.cancel()
        try:
            rec = await asyncio.wait_for(task, timeout=45)
        except asyncio.TimeoutError:
            rec = relay.live.pop(rid)
            rec.status, rec.outcome = "timeout", "gave_up_after_start"
            rec.total_ms = 45000.0
        relay.live.pop(rid, None)
        rows.append(write(rec, relay, f"{args.tag};patience={patience:.1f}"))

    await asyncio.gather(*(user(i, at, q) for i, (at, q) in enumerate(arrivals)))
    peak = relay.peak_queue()
    await relay.aclose()
    with (RUNS / "peaks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario": args.scenario, "source": args.source, "tag": args.tag,
                            "peak_queue": peak, "requests": len(rows),
                            "duration_s": round(arrivals[-1][0], 1)}) + "\n")
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(f"done: {st} peak_queue={peak}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "load"])
    ap.add_argument("--source", choices=["real", "stand-in"], default="stand-in")
    ap.add_argument("--setting", default="320,5,0")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="")
    ap.add_argument("--understand", default="ambiguous", choices=["always", "ambiguous", "never"])
    args = ap.parse_args()
    asyncio.run(run_eval(args) if args.mode == "eval" else run_load(args))


if __name__ == "__main__":
    main()

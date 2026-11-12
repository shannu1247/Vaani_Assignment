"""Build results/results.csv from the raw run records and compute every
table in the write-up from results.csv alone.

  python -m relay.analyze            # rebuild CSV, print tables (markdown)

results/runs/*.jsonl is the raw record of every request (timeline, calls,
answer). results.csv is regenerated from it, re-grading each answer with the
current grader, so the CSV and the raw records can never disagree. Then the
tables are computed from results.csv only, as a reviewer would.

Definitions (also stated in the write-up):
  first token  = first verified answer text shown to the user (status lines excluded)
  first-token and total-time percentiles use requests with status ok (an answer
  or a decided refusal); a fast "busy" or "daily budget" reply is not counted as
  a first token, so shedding cannot make latency look better
  peak queue   = max number of requests in the system at once, from
                 start_time + total_ms
  cost         = tokens x gpt-oss-120b list price ($0.15/M in, $0.60/M out);
                 the true model mix is reported separately from runs/*.jsonl
"""
from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .grade import grade, page_match
from .retrieval import get_index

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CSV = RES / "results.csv"
COLS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
        "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
PRICE = {"openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30),
         "qwen/qwen3.8-27b": (0.80, 4.00)}
SKIP = {"smoke"}


def cited_chunks(r: dict) -> list:
    size = int(r["retrieval_setting"].split("_")[0][1:])
    by_id = {c.cid: c for c in get_index(size).chunks}
    return [(by_id[i].short, by_id[i].page_from, by_id[i].page_to) for i in r.get("cited", []) if i in by_id]


def raw_records() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RES / "runs" / "*.jsonl"))):
        if Path(f).stem in SKIP | {"peaks"}:
            continue
        for l in open(f, encoding="utf-8"):
            out.append(json.loads(l))
    return out


def rebuild_csv() -> list[dict]:
    rows = []
    for r in raw_records():
        g = grade(r["question"], r["outcome"], r["answer"], cited_chunks(r))
        rows.append({"request_id": r["request_id"], "scenario": r["scenario"], "source": r["source"],
                     "start_time": r["start_time"],
                     "first_token_ms": "" if r["first_token_ms"] is None else round(r["first_token_ms"], 1),
                     "total_ms": round(r["total_ms"], 1), "status": r["status"],
                     "retries": r["retries"], "tokens_in": r["tokens_in"],
                     "tokens_out": r["tokens_out"], "retrieval_setting": r["retrieval_setting"],
                     "answer_grade": g})
    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ from CSV only
def load_csv() -> list[dict]:
    return list(csv.DictReader(CSV.open(encoding="utf-8")))


def pct(xs, ps=(50, 95, 99)) -> str:
    xs = [float(x) for x in xs if x not in ("", None)]
    if not xs:
        return "n/a"
    return " / ".join(f"{np.percentile(xs, p):.0f}" for p in ps)


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def peak_in_system(rows) -> int:
    ev = []
    for r in rows:
        t0 = ts(r["start_time"])
        ev += [(t0, 1), (t0 + timedelta(milliseconds=float(r["total_ms"])), -1)]
    cur = peak = 0
    for _, d in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    return peak


def table2(rows) -> tuple[str, str]:
    order = ["real_steady", "real_burst", "steady", "burst", "overload", "rpm_halved",
             "daily_exhausted", "naive_retry_overload"]
    groups: dict[tuple, list] = {}
    for r in rows:
        if r["scenario"].startswith("eval_"):
            continue
        groups.setdefault((r["scenario"], r["source"]), []).append(r)
    keys = sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))
    a = ["| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) "
         "| First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |",
         "|---|---|---|---|---|---|---|"]
    b = ["| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request "
         "| Peak queue length | Tokens per answer | Cost per 1,000 answers |",
         "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        g = groups[k]
        n = len(g)
        starts = [ts(r["start_time"]) for r in g]
        ends = [ts(r["start_time"]) + timedelta(milliseconds=float(r["total_ms"])) for r in g]
        dur = (max(ends) - min(starts)).total_seconds()
        arr = (max(starts) - min(starts)).total_seconds()
        rate = n / max(arr, 1e-9) * 60 if n > 1 else 0
        ok = [r for r in g if r["status"] == "ok"]
        a.append(f"| {k[0]} | {k[1]} | {n} | {dur:.0f} | {rate:.1f} | "
                 f"{pct([r['first_token_ms'] for r in ok])} | {pct([r['total_ms'] for r in ok])} |")
        pc = lambda s: 100 * sum(r["status"] == s for r in g) / n
        tok = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in ok]
        cost = [int(r["tokens_in"]) * 0.15e-6 + int(r["tokens_out"]) * 0.60e-6 for r in ok]
        b.append(f"| {k[0]} ({k[1]}) | {pc('ok'):.1f} | {pc('rate_limited'):.1f} | {pc('timeout'):.1f} | "
                 f"{np.mean([int(r['retries']) for r in g]):.2f} | {peak_in_system(g)} | "
                 f"{np.mean(tok):.0f} | ${1000 * np.mean(cost):.2f} |" if ok else
                 f"| {k[0]} ({k[1]}) | {pc('ok'):.1f} | {pc('rate_limited'):.1f} | {pc('timeout'):.1f} | "
                 f"{np.mean([int(r['retries']) for r in g]):.2f} | {peak_in_system(g)} | n/a | n/a |")
    return "\n".join(a), "\n".join(b)


def table3(rows) -> str:
    out = ["| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % "
           "| Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) "
           "| Total time p50 / p95 (ms) |", "|---|---|---|---|---|---|---|---|"]
    # Table 3 rows = quality-eval scenarios (real Groq, gpt-oss-120b drafts):
    # quality_final is the production pipeline at c320_k5; eval_<setting> rows use
    # the same pipeline at other settings; *_understand ran the LLM understand step
    # on every question (shows what that step buys).
    scen = sorted({r["scenario"] for r in rows
                   if (r["scenario"].startswith("eval_") or r["scenario"] == "quality_final")
                   and r["source"] == "real"})
    for sc in scen:
        g = [r for r in rows if r["scenario"] == sc and r["source"] == "real"]
        s = g[0]["retrieval_setting"]
        n = len(g)
        c, s_, rr = (int(x) for x in [s.split("_")[0][1:], s.split("_")[1][1:], s.endswith("_rr")])
        pc = lambda grade_: 100 * sum(r["answer_grade"] == grade_ for r in g) / n
        tok = np.mean([int(r["tokens_in"]) + int(r["tokens_out"]) for r in g])
        note = (" (production)" if sc == "quality_final" else
                " + LLM understand on every question" if sc.endswith("_understand") else "")
        out.append(f"| ~{c} tokens, {s_}, {'yes' if rr else 'no'}{note} | {n} | "
                   f"{pc('correct'):.1f} | {pc('wrong'):.1f} | "
                   f"{pc('refused_correctly'):.1f} | {tok:.0f} | "
                   f"{pct([r['first_token_ms'] for r in g], (50, 95))} | "
                   f"{pct([r['total_ms'] for r in g], (50, 95))} |")
    return "\n".join(out)


def model_mix() -> str:
    """True cost per scenario using each call's model (from the raw records)."""
    agg: dict[tuple, dict] = {}
    for r in raw_records():
        k = (r["scenario"], r["source"])
        a = agg.setdefault(k, {"cost": 0.0, "n_ok": 0, "calls": {}})
        if r["status"] == "ok":
            a["n_ok"] += 1
        for c in r.get("calls", []):
            pi, po = PRICE.get(c["model"], (0.15, 0.60))
            a["cost"] += c["tokens_in"] * pi * 1e-6 + c["tokens_out"] * po * 1e-6
            a["calls"][c["model"].split("/")[-1]] = a["calls"].get(c["model"].split("/")[-1], 0) + 1
    lines = ["| Scenario | Source | Calls by model | Cost per 1,000 ok answers (model-accurate) |",
             "|---|---|---|---|"]
    for (s, src), a in sorted(agg.items()):
        lines.append(f"| {s} | {src} | {a['calls']} | "
                     f"{'$%.2f' % (1000 * a['cost'] / a['n_ok']) if a['n_ok'] else 'n/a'} |")
    return "\n".join(lines)


def main() -> None:
    rebuild_csv()
    rows = load_csv()
    print(f"results.csv: {len(rows)} rows\n")
    print("## Table 3: retrieval settings (real Groq, gpt-oss-120b drafts)\n")
    print(table3(rows))
    t2a, t2b = table2(rows)
    print("\n## Table 2a\n")
    print(t2a)
    print("\n## Table 2b\n")
    print(t2b)
    print("\n## Model mix and model-accurate cost (from runs/*.jsonl)\n")
    print(model_mix())


if __name__ == "__main__":
    main()

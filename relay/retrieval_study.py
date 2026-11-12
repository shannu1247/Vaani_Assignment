"""Retrieval-only study (no model calls, zero API budget).

For each answerable eval question: is a passage from the gold paper within
+-1 page of a gold page in the top-k ("hit@k")? And for unanswerable ones,
which paper's passages come back (contamination)? Compared across:
  * our settings, with the raw question vs the understand step's rewrites
    (taken from the recorded real runs, so no new calls);
  * a deliberately naive retriever: fixed 320-token windows over raw page
    text (sections ignored, references included, no paper/section header),
    dense-only top-5.

  python -m relay.retrieval_study
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import yaml

from .retrieval import QUERY_PREFIX, Setting, _embedder, approx_tokens, get_index

ROOT = Path(__file__).resolve().parent.parent


def hit(chunks, q) -> bool:
    return any(c[0] == q["paper"] and any(c[1] - 1 <= p <= c[2] + 1 for p in q["pages"])
               for c in chunks)


def naive_index():
    """Fixed windows over raw text, page-tagged, references included."""
    rows = [json.loads(l) for l in (ROOT / "papers" / "blocks.jsonl").open(encoding="utf-8")]
    wins, cur, pages, short = [], "", [], None
    for b in rows:
        if short and b["short"] != short and cur:
            wins.append((short, min(pages), max(pages), cur))
            cur, pages = "", []
        short = b["short"]
        cur += b["text"] + "\n"
        pages.append(b["page"])
        if approx_tokens(cur) >= 320:
            wins.append((short, min(pages), max(pages), cur))
            cur, pages = "", []
    if cur:
        wins.append((short, min(pages), max(pages), cur))
    cache = ROOT / "papers" / "index_naive.npz"
    if cache.exists() and np.load(cache)["emb"].shape[0] == len(wins):
        emb = np.load(cache)["emb"]
    else:
        emb = np.array(list(_embedder().embed([w[3] for w in wins], batch_size=16)), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.savez_compressed(cache, emb=emb)
    return wins, emb


def main() -> None:
    qs = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    # understand-step rewrites recorded during the real eval runs
    rewrites = {}
    for f in glob.glob(str(ROOT / "results" / "runs" / "eval_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            u = r.get("understanding") or {}
            if u.get("source") == "llm":
                rewrites.setdefault(r["question"], u)
    settings = [Setting(128, 8, False), Setting(320, 5, False), Setting(320, 5, True), Setting(700, 3, False)]
    ans = [q for q in qs if q["answerable"]]
    una = [q for q in qs if not q["answerable"]]
    print(f"{len(ans)} answerable questions, rewrites recorded for {sum(q['q'] in rewrites for q in ans)}\n")
    print("| Retriever | hit@k raw question | hit@k with understand rewrites | median retrieve ms |")
    print("|---|---|---|---|")
    detail = {}
    import time
    for st in settings:
        ix = get_index(st.chunk_tokens)
        res = {"raw": 0, "rw": 0}
        ms = []
        for q in ans:
            t = time.perf_counter()
            h, _ = ix.search([q["q"]], st)
            ms.append((time.perf_counter() - t) * 1000)
            ok_raw = hit([(c.short, c.page_from, c.page_to) for c, _ in h], q)
            res["raw"] += ok_raw
            u = rewrites.get(q["q"])
            if u:
                h2, _ = ix.search([q["q"]] + [s for s in u.get("queries", []) if s != q["q"]], st,
                                  u.get("papers") or None)
                res["rw"] += hit([(c.short, c.page_from, c.page_to) for c, _ in h2], q)
            detail.setdefault(q["id"], {})[st.name] = ok_raw
        print(f"| {st.name} | {res['raw']}/{len(ans)} | {res['rw']}/{len(ans)} | {np.median(ms):.0f} |")

    wins, emb = naive_index()
    nh = 0
    naive_rows = []
    for q in ans + una:
        v = np.array(list(_embedder().embed([QUERY_PREFIX + q["q"]])), dtype=np.float32)[0]
        v /= np.linalg.norm(v)
        top = np.argsort(-(emb @ v))[:5]
        got = [wins[i] for i in top]
        if q["answerable"]:
            ok = hit([(w[0], w[1], w[2]) for w in got], q)
            nh += ok
        else:
            ok = None
        naive_rows.append((q["id"], ok, [(w[0], w[1], w[2]) for w in got],
                           [w[3][:90].replace("\n", " ") for w in got]))
    print(f"| naive (fixed 320, refs in, no header, dense top-5) | {nh}/{len(ans)} | - | - |\n")

    print("Per-question hit (raw question):")
    for qid, d in detail.items():
        print(f"  {qid}: " + "  ".join(f"{k}={'Y' if v else '.'}" for k, v in d.items()))
    print("\nNaive retriever top-5 for selected questions:")
    for qid, ok, got, txt in naive_rows:
        if qid in ("q10", "q14", "u01", "u04", "q07", "q18", "u02"):
            print(f"  {qid} hit={ok}")
            for g, t in zip(got, txt):
                print(f"     {g}  {t!r}")


if __name__ == "__main__":
    main()

"""Retrieval tool: section-bounded chunks, hybrid BM25 + dense search fused
with reciprocal-rank fusion, optional cross-encoder re-rank.

Design choices (each is a defence against a way this library misleads a
naive retriever, see WRITEUP):
  * chunks never cross a section boundary, so a citation names one section;
  * each chunk is indexed with a context header "<paper> §<num> <title>" so
    "attention" in Moshi (RQ-Transformer, Mimi) is separable from "attention"
    in the Transformer paper, and H-Net "chunking" from generic chunking;
  * reference lists are not indexed (titles of cited papers match queries
    but answer nothing);
  * table captions are kept with the rows that follow them.

A retrieval *setting* = (chunk_tokens, k, rerank). Indexes are built once per
chunk size and cached in papers/index_<size>.npz.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = ROOT / "papers" / "blocks.jsonl"
EMB_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # CPU: L-12 took 3.8 s for 24; L-6 for 12 = 0.57 s
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")
STOP = set("""a an the of to in on for and or is are was were be by with as at from that this
these those it its which what how why when who whom does do did can could would should into
than then there their them they we our you your i me my about over under between not no
paper papers model models use used using""".split())


def tokens(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOP]


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


@dataclass(frozen=True)
class Setting:
    chunk_tokens: int
    k: int
    rerank: bool
    pool: int = 12

    @property
    def name(self) -> str:
        return f"c{self.chunk_tokens}_k{self.k}_{'rr' if self.rerank else 'norr'}"


@dataclass
class Chunk:
    cid: str
    short: str
    paper_id: str
    sec_num: str
    sec_title: str
    page_from: int
    page_to: int
    text: str

    @property
    def cite(self) -> str:
        pages = f"p.{self.page_from}" if self.page_from == self.page_to \
            else f"pp.{self.page_from}-{self.page_to}"
        sec = f"§{self.sec_num} {self.sec_title}" if self.sec_num else self.sec_title
        return f"{self.short}, {sec}, {pages}"

    @property
    def header(self) -> str:
        return f"{self.short} — {self.sec_title} (§{self.sec_num})" if self.sec_num \
            else f"{self.short} — {self.sec_title}"


def build_chunks(chunk_tokens: int) -> list[Chunk]:
    raw = [json.loads(l) for l in BLOCKS.open(encoding="utf-8")]
    raw = [b for b in raw if b["kind"] != "references"]
    # split blocks bigger than the target at sentence boundaries
    blocks = []
    for b in raw:
        if approx_tokens(b["text"]) <= chunk_tokens:
            blocks.append(b)
            continue
        part = ""
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", b["text"]):
            if part and approx_tokens(part + s) > chunk_tokens:
                blocks.append({**b, "text": part.strip()})
                part = ""
            part += s + " "
        if part.strip():
            blocks.append({**b, "text": part.strip()})
    chunks: list[Chunk] = []
    cur: list[dict] = []

    def flush():
        if not cur:
            return
        b0 = cur[0]
        n = sum(1 for c in chunks if c.short == b0["short"])
        chunks.append(Chunk(
            cid=f"{b0['short'][0]}{n}", short=b0["short"], paper_id=b0["paper_id"],
            sec_num=b0["sec_num"], sec_title=b0["sec_title"],
            page_from=min(b["page"] for b in cur), page_to=max(b["page"] for b in cur),
            text="\n".join(b["text"] for b in cur)))

    for b in blocks:
        same = cur and (b["short"], b["sec_num"], b["sec_title"]) == \
            (cur[0]["short"], cur[0]["sec_num"], cur[0]["sec_title"])
        size = sum(approx_tokens(x["text"]) for x in cur)
        if not same or size + approx_tokens(b["text"]) > chunk_tokens:
            # keep a caption attached to what follows it
            carry = [cur.pop()] if same and cur and cur[-1]["kind"] == "caption" else []
            flush()
            cur = carry
        cur.append(b)
    flush()
    # merge tiny tails (e.g. a 1-line block left alone) into the previous chunk of the same section
    merged: list[Chunk] = []
    for c in chunks:
        if merged and approx_tokens(c.text) < chunk_tokens // 5 and \
                (merged[-1].short, merged[-1].sec_num) == (c.short, c.sec_num):
            p = merged[-1]
            p.text += "\n" + c.text
            p.page_to = max(p.page_to, c.page_to)
        else:
            merged.append(c)
    for i, c in enumerate(merged):            # stable, readable ids: T12, M140, H77
        c.cid = f"{c.short[0]}{i}"
    return merged


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMB_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


class Index:
    def __init__(self, chunk_tokens: int):
        self.chunk_tokens = chunk_tokens
        self.chunks = build_chunks(chunk_tokens)
        docs = [f"{c.header}\n{c.text}" for c in self.chunks]
        self.bm25 = BM25Okapi([tokens(d) for d in docs])
        cache = ROOT / "papers" / f"index_{chunk_tokens}.npz"
        if cache.exists() and np.load(cache)["emb"].shape[0] == len(docs):
            self.emb = np.load(cache)["emb"]
        else:
            self.emb = np.array(list(_embedder().embed(docs, batch_size=16)), dtype=np.float32)
            self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
            np.savez_compressed(cache, emb=self.emb)

    def search(self, queries: list[str], setting: Setting, papers: list[str] | None = None
               ) -> tuple[list[tuple[Chunk, float]], dict]:
        """Hybrid search for one or more query phrasings. Returns (chunk, score)
        best first and timing info. score is the cross-encoder logit when
        rerank is on, else the fused RRF score."""
        t0 = time.perf_counter()
        fused: dict[int, float] = {}
        qv = np.array(list(_embedder().embed([QUERY_PREFIX + q for q in queries])),
                      dtype=np.float32)
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        for qi, q in enumerate(queries):
            bm = self.bm25.get_scores(tokens(q))
            dn = self.emb @ qv[qi]
            for ranking in (np.argsort(-bm)[:50], np.argsort(-dn)[:50]):
                for r, i in enumerate(ranking):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r)
        if papers:   # soft paper preference from the understand step, never a hard filter
            for i in fused:
                if self.chunks[i].short in papers:
                    fused[i] *= 1.25
        order = sorted(fused, key=fused.get, reverse=True)
        t_fuse = time.perf_counter()
        if setting.rerank:
            pool = order[:setting.pool]
            q = queries[0]
            scores = list(_reranker().rerank(
                q, [f"{self.chunks[i].header}\n{self.chunks[i].text}" for i in pool]))
            ranked = sorted(zip(pool, scores), key=lambda x: -x[1])[:setting.k]
        else:
            ranked = [(i, fused[i]) for i in order[:setting.k]]
        info = {"fuse_ms": round((t_fuse - t0) * 1000, 1),
                "rerank_ms": round((time.perf_counter() - t_fuse) * 1000, 1)}
        return [(self.chunks[i], float(s)) for i, s in ranked], info


_INDEXES: dict[int, Index] = {}


def get_index(chunk_tokens: int) -> Index:
    if chunk_tokens not in _INDEXES:
        _INDEXES[chunk_tokens] = Index(chunk_tokens)
    return _INDEXES[chunk_tokens]


if __name__ == "__main__":
    import sys
    for size in (128, 320, 700):
        ix = get_index(size)
        lens = [approx_tokens(c.text) for c in ix.chunks]
        print(f"size {size}: {len(ix.chunks)} chunks, median {int(np.median(lens))} tok, "
              f"max {max(lens)}")
    q = sys.argv[1] if len(sys.argv) > 1 else "How many codebooks does Mimi use?"
    for st in (Setting(320, 5, False), Setting(320, 5, True)):
        hits, info = get_index(st.chunk_tokens).search([q], st)
        print(f"\n{st.name} {info}")
        for c, s in hits:
            print(f"  {s:7.3f} [{c.cid}] {c.cite} :: {c.text[:110]!r}")

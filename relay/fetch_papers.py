"""Download the three library papers at their exact arXiv versions and extract
text blocks tagged with page and section.

Sections come from the PDF's own outline (hyperref bookmarks), positioned on
the page by their link target, so a block is assigned to the section it is
really in even when a heading starts mid-page. Section numbers are rebuilt
from the outline hierarchy (1, 3.2.1, ...); level-1 entries after
"Conclusion" are appendices and get letters (A, B, ...).

Output: papers/pdf/<id><v>.pdf and papers/blocks.jsonl, one row per block:
  paper_id, version, short, title, page, sec_num, sec_title, kind, text
kind is "body", "caption", "references" or "front".

Run:  .venv\\Scripts\\python -m relay.fetch_papers
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "papers" / "pdf"
BLOCKS = ROOT / "papers" / "blocks.jsonl"

PAPERS = [
    {"arxiv": "1706.03762", "version": "v7", "short": "Transformer",
     "title": "Attention Is All You Need"},
    {"arxiv": "2410.00037", "version": "v1", "short": "Moshi",
     "title": "Moshi: a speech-text foundation model for real-time dialogue"},
    {"arxiv": "2507.07955", "version": "v2", "short": "H-Net",
     "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"},
]

CAPTION = re.compile(r"^(Table|Figure|Fig\.)\s+\d+[:.]")


def download(p: dict) -> Path:
    out = PDF_DIR / f"{p['arxiv']}{p['version']}.pdf"
    if out.exists() and out.stat().st_size > 100_000:
        return out
    url = f"https://arxiv.org/pdf/{p['arxiv']}{p['version']}"
    r = httpx.get(url, follow_redirects=True, timeout=120,
                  headers={"User-Agent": "relay-assignment/0.1"})
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF")
    out.write_bytes(r.content)
    return out


def outline(doc) -> list[dict]:
    """Outline entries with rebuilt numbers and (page, y) start positions."""
    toc = doc.get_toc(simple=False)
    counters = [0, 0, 0, 0]
    appendix, letter = False, 0
    out = []
    for level, title, page, dest in toc:
        title = " ".join(title.split())
        if level == 1:
            if appendix:
                letter += 1
            counters = [counters[0] + 1, 0, 0, 0]
        else:
            counters[level - 1] += 1
            for k in range(level, 4):
                counters[k] = 0
        head = chr(ord("A") + letter - 1) if appendix and level >= 1 and letter else str(counters[0])
        num = ".".join([head] + [str(c) for c in counters[1:level]])
        y = 0.0
        to = dest.get("to") if isinstance(dest, dict) else None
        if to is not None:
            # outline points use PDF coordinates (origin bottom-left)
            y = doc[page - 1].rect.height - to.y
        out.append({"num": num, "title": title, "page": page, "y": max(y - 2, 0)})
        if level == 1 and title.lower().startswith("conclusion") and not appendix:
            appendix = True
    return out


def block_text(block: dict) -> str:
    """Rebuild a block's text from spans, keeping superscripts as ^x so that
    '1.0 · 10^20' does not collapse into '1.0 · 1020' (a silent wrong number)."""
    lines = []
    for line in block.get("lines", []):
        out = ""
        for s in line["spans"]:
            t = s["text"]
            if s["flags"] & 1 and t.strip() and out and not out.endswith(" "):
                t = "^" + t.strip()
            out += t
        lines.append(out)
    return "\n".join(lines)


def extract(p: dict, pdf: Path) -> list[dict]:
    doc = pymupdf.open(pdf)
    secs = outline(doc)
    rows = []
    in_refs = False
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", sort=True)["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            y0 = b["bbox"][1]
            text = block_text(b)
            text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate
            text = " ".join(text.split())
            if len(text) < 3 or re.fullmatch(r"\d{1,3}", text):   # page numbers
                continue
            # current section = last outline entry starting at or before this block
            cur = None
            for s in secs:
                if (s["page"], s["y"]) <= (pno, y0 + 1):
                    cur = s
            # References are not in the outline: detect the heading, end at next outline entry
            if re.fullmatch(r"References|Bibliography", text):
                in_refs = True
                ref_start = (pno, y0)
                continue
            if in_refs and cur is not None and (cur["page"], cur["y"]) > ref_start:
                in_refs = False
            # Transformer v7 has an un-bookmarked "Attention Visualizations" appendix
            # after the references: keep its figure captions, drop the per-word labels.
            if in_refs and text == "Attention Visualizations":
                in_refs, cur = False, {"num": "A", "title": text, "page": pno, "y": y0}
                secs.append(cur)
                continue
            if cur is not None and cur["title"] == "Attention Visualizations":
                if not CAPTION.match(text):
                    continue
            if in_refs:
                kind, num, title = "references", "", "References"
            elif cur is None:
                kind, num, title = "front", "", "Abstract / front matter"
            else:
                kind = "caption" if CAPTION.match(text) else "body"
                num, title = cur["num"], cur["title"]
            # skip the heading line itself (it is metadata now)
            if cur is not None and text.lstrip("0123456789. ").startswith(cur["title"][:25]) \
                    and len(text) < len(cur["title"]) + 12:
                continue
            rows.append({"paper_id": p["arxiv"], "version": p["version"],
                         "short": p["short"], "title": p["title"], "page": pno,
                         "sec_num": num, "sec_title": title, "kind": kind,
                         "text": text})
    return rows


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for p in PAPERS:
        pdf = download(p)
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
        rows = extract(p, pdf)
        kinds = {k: sum(r["kind"] == k for r in rows) for k in ("body", "caption", "references", "front")}
        n_sec = len({(r["sec_num"], r["sec_title"]) for r in rows})
        print(f"{p['short']:<12} {p['arxiv']}{p['version']}  blocks={len(rows):>4}  "
              f"sections={n_sec:>3}  {kinds}  sha256={sha}")
        all_rows += rows
    with BLOCKS.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {BLOCKS} ({len(all_rows)} blocks)")


if __name__ == "__main__":
    main()

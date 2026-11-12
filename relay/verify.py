"""Deterministic answer checker, applied sentence by sentence while the draft
streams, so verified sentences can be released immediately and an
unsupported one is never shown.

A sentence passes when:
  * every citation id it carries was actually retrieved for this request;
  * it carries at least one citation (short connective sentences excepted);
  * every number in it appears in at least one of the passages it cites
    (the cheapest strong check against invented or cross-paper values).

This is code, not a model call: a judge model would double the per-answer
token cost on the binding limit (TPM) and add ~0.8 s.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = re.compile(r"\[\s*[A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*\s*\]")
CITE_ID = re.compile(r"[A-Z]\d+")
# Names that belong to exactly one paper. A sentence that names one must cite
# that paper (catches "Moshi scores 41.8 BLEU [T30]": right number, wrong paper).
# "Transformer" is not here: Moshi (RQ-Transformer) and H-Net both use the word.
OWNED = {"M": re.compile(r"\b(Moshi|Mimi|Helium|Inner Monologue|RQ-Transformer)\b", re.I),
         "H": re.compile(r"\b(H-Net|dynamic chunking|routing module|smoothing module|ratio loss)\b", re.I)}
NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)(?:\s*(%|×|x))?")
REFUSAL = re.compile(r"NOT_IN_PAPERS|(do(es)? not|don't|doesn't) (say|state|mention|report|specify|provide|clarify|indicate|give|disclose)"
                     r"|not (stated|reported|specified|mentioned|given|disclosed|provided)|no information", re.I)


def clean_text(s: str) -> str:
    """Models emit narrow no-break spaces and non-breaking hyphens; normalise
    them so users, the verifier and the grader all see plain text."""
    # "2 048" / "2 048" (thin-space thousands separator) -> "2048"
    s = re.sub(r"(\d)[    ](\d{3})(?!\d)", r"\1\2", s)
    return (s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
             .replace("‑", "-").replace("‐", "-")
             # gpt-oss sometimes cites with its native full-width brackets 【M107】
             .replace("【", "[").replace("】", "]"))
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-(])|\n{2,}|\n(?=[-*•] )")


def norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    cites: list[str] = field(default_factory=list)


def check_sentence(sent: str, passages: dict[str, str]) -> Verdict:
    ids = [c for m in CITE.finditer(sent) for c in CITE_ID.findall(m.group(0))]
    if not ids:
        words = len(sent.split())
        if REFUSAL.search(sent) or (words <= 8 and not NUM.search(sent)):
            return Verdict(True, "no-claim")
        return Verdict(False, "uncited claim")
    unknown = [c for c in ids if c not in passages]
    if unknown:
        return Verdict(False, f"cites unretrieved {unknown}", ids)
    clean_text = CITE.sub("", sent)
    for paper, pat in OWNED.items():
        if pat.search(clean_text) and not any(c.startswith(paper) for c in ids) \
                and not any(p.search(clean_text) for k, p in OWNED.items() if k != paper):
            return Verdict(False, f"names a {paper}-paper concept but cites {ids}", ids)
    body = " ".join(passages[c] for c in ids)
    body_nums = {norm_num(n) for n, _ in NUM.findall(body)}
    clean = CITE.sub("", sent)
    for n, _ in NUM.findall(clean):
        v = norm_num(n)
        if len(v) <= 1:                      # single digits: too common to check
            continue
        if v not in body_nums and v not in body.replace(",", ""):
            return Verdict(False, f"number {n} not in cited passages", ids)
    return Verdict(True, "", ids)


class StreamVerifier:
    """Feed streamed text; get back sentences that passed, in order."""

    def __init__(self, passages: dict[str, str]):
        self.passages = passages
        self.buf = ""
        self.released: list[str] = []
        self.rejected: list[tuple[str, str]] = []
        self.cited: set[str] = set()
        self.held: list[str] = []      # uncited sentences wait for a cited one
        self.repaired: list[tuple[str, str]] = []

    def _repair(self, sent: str) -> str:
        """[P28] when T28 was retrieved: accept only if exactly one retrieved id
        has that number (seen on real Groq: a correct answer cited [P28], was
        rejected, and the deterministic redraft repeated it)."""
        def fix(m):
            out = []
            for i in CITE_ID.findall(m.group(0)):
                if i not in self.passages:
                    cand = [k for k in self.passages if k[1:] == i[1:]]
                    if len(cand) == 1:
                        self.repaired.append((i, cand[0]))
                        i = cand[0]
                out.append(i)
            return "[" + ", ".join(out) + "]"
        return CITE.sub(fix, sent)

    def feed(self, piece: str) -> list[str]:
        self.buf += clean_text(piece)
        parts = SENT_END.split(self.buf)
        if len(parts) == 1:
            return []
        self.buf = parts[-1]
        out: list[str] = []
        for p in parts[:-1]:
            out += self._judge(p)
        return out

    def finish(self) -> list[str]:
        tail, self.buf = self.buf, ""
        out = self._judge(tail)
        if self.cited and self.held:   # trailing caveats after a cited answer
            out += self.held
            self.released += self.held
            self.held = []
        return out

    def _judge(self, sent: str) -> list[str]:
        """Uncited sentences ("the papers do not give X") are only shown once
        the answer has at least one verified, cited sentence; otherwise a draft
        that later turns out unverifiable would already have leaked text."""
        sent = sent.strip()
        if not sent or "NOT_IN_PAPERS" in sent:
            return []
        sent = self._repair(sent)
        v = check_sentence(sent, self.passages)
        if not v.ok:
            self.rejected.append((sent, v.reason))
            return []
        if not v.cites:
            if self.cited:
                self.released.append(sent)
                return [sent]
            self.held.append(sent)
            return []
        out = self.held + [sent]
        self.held = []
        self.released += out
        self.cited.update(v.cites)
        return out

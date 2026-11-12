"""Automatic grading against eval/questions.yaml (rules documented there).

Every automatic grade is also written next to the answer in the run file so
it can be audited by hand; manual overrides go in eval/grade_overrides.yaml
({request_id: grade, reason}) and are applied on top.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from .verify import REFUSAL, clean_text

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def questions() -> dict[str, dict]:
    qs = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    return {q["q"]: q for q in qs}


def grade(question: str, outcome: str, answer: str, cited_chunks: list) -> str:
    """cited_chunks: [(short, page_from, page_to)] for every passage the
    released answer cites."""
    g = questions().get(question)
    if g is None:
        return ""
    answer = clean_text(answer)
    # Only a refusal the pipeline DECIDED on ("the papers do not say") can be a
    # correct refusal. A failure, a shed ("busy"/"daily_exhausted") or a user
    # who gave up is not knowledge, so it is graded refused_wrongly even for an
    # unanswerable question (conservative; found when 3 failed calls during a
    # network outage were being counted as correct refusals).
    if outcome not in ("answered", "refused"):
        return "refused_wrongly"
    refused = outcome != "answered"
    if not g["answerable"]:
        if refused:
            return "refused_correctly"
        # an answer that says the papers don't state it (possibly with context) is a correct refusal
        return "refused_correctly" if REFUSAL.search(answer) else "wrong"
    if refused:
        return "refused_wrongly"
    facts_ok = all(any(re.search(p, answer, re.I) for p in group) for group in g["must"])
    # The right paper must be cited. Pages are not required: papers restate
    # facts in several places (Moshi gives the acoustic delay in Table 1 and
    # again in §5.3); page_match() reports page agreement separately.
    cite_ok = any(s == g["paper"] for s, _, _ in cited_chunks)
    return "correct" if facts_ok and cite_ok else "wrong"


def page_match(question: str, cited_chunks: list) -> bool | None:
    g = questions().get(question)
    if not g or not g.get("pages"):
        return None
    return any(s == g["paper"] and any(pf - 1 <= p <= pt + 1 for p in g["pages"])
               for s, pf, pt in cited_chunks)

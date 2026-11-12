# The Relay — cited Q&A over three papers, built to survive launch day

A small, working version of a multi-step answering pipeline over three arXiv
papers, a measured model of the real provider (Groq free tier), a stand-in
model service calibrated from those measurements, and load runs under five
shapes. The write-up is in `WRITEUP.md`; every number in it is recomputed
from `results/results.csv` by `python -m relay.analyze`.

## Layout

| Path | What it is |
|---|---|
| `relay/fetch_papers.py` | Downloads 1706.03762**v7**, 2410.00037**v1**, 2507.07955**v2**; extracts text blocks tagged with page and section (from the PDF outline), keeps superscripts (`10^19`), drops reference lists from the index |
| `relay/retrieval.py` | Section-bounded chunks with a "paper — section" header, hybrid BM25 + dense (bge-base) with reciprocal-rank fusion, optional cross-encoder re-rank |
| `relay/pipeline.py` | The pipeline: cache/coalesce → understand (LLM) → retrieve (tool) + table lookup (tool) → decide (branch) → draft (LLM, streamed) → verify (per sentence) → bounded loop |
| `relay/verify.py` | Sentence verifier: citations must be to retrieved passages, numbers must appear in the cited passage, paper-specific names must cite that paper |
| `relay/governor.py` | Client-side mirror of the provider's limits (RPM/TPM/RPD buckets), deadline-aware admission, re-sync from headers and 429s |
| `relay/llm.py` | One streaming client for Groq and the stand-in; records TTFT, total, usage, Groq's server-side queue/prompt/completion split, rate-limit headers |
| `relay/groq_probe.py` | The measurements of the real service (`results/probe/*.jsonl`) |
| `relay/standin.py` | The stand-in model service (FastAPI, SSE), calibrated from the probe data |
| `relay/run.py` | Quality evaluation and load scenarios; writes `results/runs/*.jsonl` |
| `relay/analyze.py` | Rebuilds `results/results.csv` from the raw runs and prints every table |
| `relay/retrieval_study.py` | Retrieval-only study (no API calls), incl. a naive retriever baseline |
| `eval/questions.yaml` | 26 evaluation questions (20 answerable, 6 not in the papers) with gold facts and pages |

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
echo GROQ_API_KEY=... > .env                       # never committed (.gitignore)

.venv\Scripts\python -m relay.fetch_papers         # papers + blocks.jsonl
.venv\Scripts\python -m relay.groq_probe headers   # then: latency / rpm / tpm (see docstring)
.venv\Scripts\python -m relay.run eval --source real --setting 320,5,0

.venv\Scripts\python -m relay.standin --port 8088  # in another terminal
.venv\Scripts\python -m relay.run load --source stand-in --scenario burst
.venv\Scripts\python -m relay.analyze              # results.csv + all tables
```

The API key is read from `.env` only, never logged; Groq's 429 bodies contain
the organization id, which is redacted (`org_REDACTED`) before anything is written.

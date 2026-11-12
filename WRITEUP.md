# The Relay — write-up

> DRAFT. Values marked ⟦TBD⟧ come from the day-2 runs (`scripts/day2_real_runs.sh`);
> every table is printed by `python -m relay.analyze` from `results/results.csv`.

## 1. What I built and why
Python pipeline (`relay/`): **cache/coalesce → understand → retrieve (tool) + table lookup (tool) → decide (branch) → draft (Groq, streamed) → verify (code, per sentence) → one bounded re-retrieval loop.**
"Fast" = first *verified* sentence on screen within 2 s at p95 for normal load; nothing unverified is ever shown.
**The decision that shaped it most:** I measured that Groq's free tier charges *prompt + max_tokens* against 8,000 tokens/min **before** the call runs, and 200,000 tokens/day per model. A 1,900-token answer therefore costs ~1/4 of a model's minute and 1/105 of its day: tokens, not requests, are the scarce resource. So the client keeps its own mirror of every limit (`governor.py`), computes waits before sending instead of discovering 429s, keeps prompts small (5 × ~260-token pieces, max_tokens 520), and treats the three models' separate budgets as a fallback chain (gpt-oss-120b → gpt-oss-20b → qwen3.8-27b).

## 2. The real service and my stand-in
How I found the limits: 3 tiny calls to read headers, 54 paced realistic calls for latency, a 45-request burst for RPM, back-to-back 2.6k-token prompts for TPM, and the day's real usage for TPD (`results/probe/`). All four limits are **continuously refilling buckets**, not windows: after one request `reset-requests` was 1m26.4s (= 86,400 s / 1,000); after the burst the next call succeeded 662 ms later; the TPD 429 asked for 286 s for 661 tokens over (= 2.315 tok/s = 200,000/day).
**Stand-in** (`standin.py`, FastAPI SSE): latency = network RTT (lognormal, ~390 ms, all before first token, measured as ttft − queue − prompt_time) + server queue (per-model lognormal from Groq's `x_groq.queue_time`) + prompt at measured tok/s + streamed decode at measured tok/s (477/868/515 for 120b/20b/qwen; Groq publishes 500/1000/450). Queue grows by ~120 ms per request in flight beyond 8; same four buckets and byte-compatible 429 bodies with `retry-after`; 1% 5xx, 0.5% mid-stream drops, 3% "slow success" stalls of 4–12 s. Replies replay real Groq text recorded for the same prompt (22/26 eval prompts covered on day 1).
**Least like the real thing:** (1) it cannot get worse *for us* when *other tenants* are busy — the real 120b queue time ranged 150–1,670 ms independently of our load; (2) a fallback model replays the primary's text, so on the stand-in a "weaker answer" is not actually weaker; (3) calibration check: stand-in TTFT p50 736 ms vs real 807 ms (9% fast), p90 987 vs 1,206 ms — its tail is too light.

**Table 1: Limits found in the real service (per model; all three chat models identical)**

| Limit | Value found | How found | Confidence |
|---|---|---|---|
| Requests per minute | 30 | 45 concurrent tiny calls: 31 ok, 14 × 429 "RPM: Limit 30" | high |
| Tokens per minute | 8,000, charged as prompt + max_tokens at admission | headers (`limit-tokens`), remaining drop 142 = 78 prompt + 64 max_tokens; 429 "TPM: Limit 8000 … Requested 2581" | high |
| Requests per day | 1,000 | `x-ratelimit-limit-requests`; refill 1 per 86.4 s | high (header), not hit |
| Tokens per day | 200,000 (actual use) | hit for real: 429 "TPD: Limit 200000, Used 198853, Requested 1808" | high |
| Requests in flight at once | no cap seen up to 31 | 31 simultaneous calls accepted; every refusal named RPM | medium |
| What it returns at a limit | HTTP 429, JSON `rate_limit_exceeded` naming the limit, its value, used and requested; `retry-after` (s, rounded up); body includes the org id (redacted in logs) | probe logs | high |
| Time until it recovers | RPM: 2 s per request (first retry ok after 662 ms); TPM: deficit ÷ 133 tok/s (14.4 s asked, ok after); TPD: deficit ÷ 2.315 tok/s (286 s for 661 tokens) — a 1.9k-token answer every ~14 min per model once the day is spent | timed recovery polls, retry-after arithmetic | high |

## 3. Results
Real runs: `real_*`, `quality_final`, `eval_*`. Stand-in: `steady`, `burst`, `overload`, `rpm_halved`, `daily_exhausted`, `naive_retry_overload`. First token = first verified sentence shown; p50/p95/p99 over requests that showed text; total time over status ok. Peak queue = max requests in the system at once (from start_time + total_ms). Cost = tokens × gpt-oss-120b list price ($0.15/M in, $0.60/M out); model-accurate cost in the last table.
⟦TBD tables 2a, 2b from relay.analyze⟧
**A number you could not have guessed:** the free tier's binding daily limit is tokens, not requests — about **105 answers per model per day** (200,000 / ~1,900), not the 1,000 the request limit suggests.
Answer quality (production, 26 own questions, 20 answerable / 6 not in the papers): ⟦TBD⟧ correct and cited; tokens per answer ⟦TBD⟧.

## 4. Where the time goes
⟦TBD: one slow request from real_burst/real_steady split into wait / work / retry with its timeline (results/runs/*.jsonl `timeline`), and whether the delay was before or after first token.⟧

## 5. When a limit is hit
**Per-minute:** the governor knows each model's minute buckets, so it does not send a call it knows will be refused. A request may wait **≤ 1 s** for the strongest model; otherwise it goes to whichever model can start soonest; if none can start within the 6 s first-token budget it is **dropped up front** with a "busy" reply that still lists the relevant passages, rather than queued past the user's patience. Requests already past retrieval get priority over new ones. Identical questions in flight share one draft; answered ones come from the cache instantly.
**Per-day:** the governor mirrors tokens/day too and re-syncs from the 429 ("Used 198853"); when a model's day is spent its traffic moves to the next model (real run `real_primary_day_exhausted`: one 429, then 0; 20/21 served by 20b or qwen). When all three are spent: cached answers still served; new questions get the search results and an honest refill time (section 7).
Who is delayed: at most 1 s, people whose question can use the best model. Who is dropped: new arrivals when nothing can start within 6 s. Who gets a weaker answer: anyone arriving while 120b's minute or day budget is spent (still verified, and labelled in the record).

## Retrieval: quality versus cost
⟦TBD Table 3⟧ Offline (no API calls, `retrieval_study.py`), hit@k of the right page over 20 answerable questions: 128×8 20/20, 320×5 19/20, 320×5 + re-rank 17/20, 700×3 19/20. The MS-MARCO cross-encoder made retrieval *worse* on paper text and cost 0.57–2.8 s on CPU. Choice: **~320-token section-bounded pieces, 5 fetched, no re-rank** ⟦confirm with Table 3⟧. Near the limit: retrieval itself is free (local), so it is never cut; what shrinks is what we *send* — the reservation is computed per call, and the draft model falls back before the user waits.

## How this library will mislead a naive retriever
1. **Same word, different paper.** "How many layers does the Transformer have?" — a naive dense retriever's top 4 are Moshi and H-Net passages (Moshi's Temporal Transformer has 32 layers); the answer in *Attention Is All You Need* is 6. Fix: paper/section header on every piece; verifier rejects a sentence naming Moshi/H-Net concepts that cites another paper.
2. **PDF extraction flattens numbers and tables.** Plain text extraction turns the big model's 2.3·10^19 FLOPs into "2.3 · 1019" and Table 3 into one cell per line, so "big … P_drop 0.3" loses its row: "What dropout did the big model use for EN-FR?" is answered 0.3; the text (§6.1) says 0.1. Fix: superscripts kept as `^`, captions kept with rows.
3. **The papers describe other people's work, and contradict themselves.** "How is chunking done?" returns H-Net's related-work appendix (static/external chunking — the methods H-Net argues against); Moshi p.61 says "under CC BY" about a borrowed table, inviting "Moshi is CC BY"; the Transformer abstract and Table 2 say 41.8 BLEU EN-FR, §6.1 says 41.0. Fix: references not indexed; the drafter must cite, and the verifier checks numbers against the cited passage — not fully solved (see 8).

## 6. Retries
Policy: never retry a call after the user has seen text; retry once on another model for connect errors, 5xx, or no first token within 4 s (measured p99 2.1 s); on 429 never resend to the same model — sync the governor from the reply and fall to the next model; a re-draft must change the prompt.
**Where retrying made it worse (real):** a correct answer cited `[P28]` instead of `[T28]`; the verifier rejected it and the loop re-drafted — at temperature 0 the identical prompt produced the identical mistake, doubling tokens and adding 0.9 s before refusing a question we had answered. Fix: repair an unambiguous id, and a re-draft now carries the rejection reason. Under overload, naive backoff-and-retry ⟦TBD naive_retry_overload vs overload numbers⟧.

## 7. What the person sees
At 3 pm all three models' daily token budgets are spent. A new question gets, within ~1 s:
> Today's budget for written answers is used up, so I can't write and check a new answer right now, and I won't guess. Search still works, and these passages look most relevant to your question:
> - Moshi, §3.3.1 Architecture, p.11: "Quantization rate. We use Q = 8 quantizers…"
> Questions already answered today are still answered instantly. The budget refills gradually rather than at midnight: the next new answer can be written in about 5 minutes, and full capacity returns over 24 hours.

The refill time is computed from the governor's mirror of the TPD bucket, not a fixed string.

## 8. What I'm unsure about, what I left out, what I learned
**Least certain assumption:** that the paper-specific regex verifier catches most wrong answers. It checks that numbers appear in the cited passage, not that they mean what the sentence says: in a real run the drafter said Moshi's practice latency is "roughly 2 seconds" (it is 200 ms; "2 s" is a TTS delay in an appendix) and it passed. If the assumption is wrong, the system is fast and confidently wrong at the same rate as the model — a judge model would cost a second call on the binding token budget.
**Deliberately not built:** (1) a semantic answer cache — "dropout of the base model" and "…of the big model" embed ~equally but differ in answer; (2) an LLM judge / always-on query rewriting — measured +1 retrieval hit on 20 questions for +0.8 s TTFT and the most expensive call per answer (qwen $0.80/$4.00 per M, a Preview model); it now runs only for ambiguous questions, which cost one answer (q04).
**New to me:** ⟦in your own words⟧

## Your feedback on Vaani
⟦your own words⟧

## How I worked with the assistant
⟦conversation export + your three prompts, the mistake you caught, the decision you overruled, what is distinctly yours⟧

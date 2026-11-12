#!/usr/bin/env bash
# Day 2: real-Groq runs that did not fit in day 1's gpt-oss-120b token budget
# (TPD 200,000/day, hit at 2026-11-12 ~10:55 UTC). Budget need on 120b:
# 13 + 26 + 26 drafts x ~1.9k tokens = ~125k, plus the real load runs.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe

# 1. finish the final production quality eval (resumes after q13)
$PY -m relay.run eval --source real --setting 320,5,0 --scenario quality_final --tag final_day2
# 2. Table 3: two more retrieval settings, same production pipeline
$PY -m relay.run eval --source real --setting 128,8,0 --tag retrieval_day2
$PY -m relay.run eval --source real --setting 320,5,1 --tag retrieval_day2
# 3. final real load runs
$PY -m relay.run load --source real --scenario real_steady --tag final_day2
$PY -m relay.run load --source real --scenario real_burst  --tag final_day2

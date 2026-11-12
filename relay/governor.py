"""Client-side rate governor: a local mirror of the provider's limits.

Measured on Groq (results/probe): per model, RPM 30, TPM 8000, RPD 1000, all
continuously refilling buckets; TPM is charged prompt_tokens + max_tokens at
admission. So the client can compute exactly how long a call must wait
instead of discovering it through a 429.

Waiters are served by priority (lower = sooner), then arrival order. A
waiter whose wait would pass its deadline is refused immediately rather than
queued (fail fast: the caller then degrades or sheds). The mirror is re-synced
from x-ratelimit-remaining-* headers and from retry-after on every 429, so if
the provider changes a limit without notice, the governor follows within one
call.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import re
import time
from dataclasses import dataclass


def parse_dur(s: str | None) -> float | None:
    """'1m26.4s' / '622ms' / '14.35s' -> seconds."""
    if not s:
        return None
    tot, ok = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", s):
        ok = True
        tot += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return tot if ok else None


class Bucket:
    def __init__(self, capacity: float, per_s: float):
        self.capacity, self.per = float(capacity), float(per_s)
        self.level, self.t = float(capacity), time.monotonic()

    @property
    def rate(self) -> float:
        return self.capacity / self.per

    def refill(self) -> None:
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now

    def wait_for(self, amount: float) -> float:
        self.refill()
        if amount > self.capacity:
            return float("inf")
        return max(0.0, (amount - self.level) / self.rate)


@dataclass(order=True)
class _Waiter:
    priority: int
    seq: int
    tokens: int
    deadline: float
    fut: asyncio.Future


class ModelGovernor:
    def __init__(self, model: str, rpm: int = 30, tpm: int = 8000, rpd: int = 1000,
                 safety: float = 0.95, tpd: int = 200_000):
        self.model = model
        self.rpm, self.tpm, self.rpd = Bucket(rpm, 60), Bucket(tpm * safety, 60), Bucket(rpd, 86400)
        # Tokens per day: measured on Groq (429 at "TPD: Limit 200000", retry-after
        # 286 s for 661 tokens over = 2.315 tok/s, i.e. a continuously refilling
        # bucket). Charged by tokens actually used, debited after each call.
        self.tpd = Bucket(tpd, 86400)
        self.blocked_until = 0.0
        self._heap: list[_Waiter] = []
        self._seq = itertools.count()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.peak_queue = 0
        self.stats = {"granted": 0, "refused_deadline": 0, "synced_429": 0}

    # ------------------------------------------------------------- public
    def queue_len(self) -> int:
        return len(self._heap)

    def estimate_wait(self, tokens: int) -> float:
        """Wait if this request joined now (ignores queued waiters ahead)."""
        w = max(self.rpm.wait_for(1), self.tpm.wait_for(tokens), self.rpd.wait_for(1),
                self.tpd.wait_for(tokens), self.blocked_until - time.monotonic())
        ahead = sum(x.tokens for x in self._heap)
        return max(w, self.tpm.wait_for(tokens + ahead))

    def day_remaining(self, answer_tokens: int = 1900) -> float:
        """Answers left today: the smaller of requests left and tokens left / answer."""
        self.rpd.refill()
        self.tpd.refill()
        return min(self.rpd.level, self.tpd.level / answer_tokens)

    def day_refill_s(self, answer_tokens: int = 1900) -> float:
        """Seconds until one more answer fits in the daily budgets."""
        return max(self.rpd.wait_for(1), self.tpd.wait_for(answer_tokens))

    def spend_day(self, tokens_used: int) -> None:
        self.tpd.refill()
        self.tpd.level -= tokens_used

    async def acquire(self, tokens: int, deadline: float, priority: int = 5) -> bool:
        """Wait for capacity for one call reserving `tokens`. False = refused
        because it could not be granted before `deadline` (monotonic s)."""
        if self.day_remaining() < 1:
            self.stats["refused_deadline"] += 1
            return False
        if time.monotonic() + self.estimate_wait(tokens) > deadline:
            self.stats["refused_deadline"] += 1
            return False
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, _Waiter(priority, next(self._seq), tokens, deadline, fut))
        self.peak_queue = max(self.peak_queue, len(self._heap))
        self._ensure_loop()
        self._wake.set()
        return await fut

    def observe(self, headers: dict, status: str, retry_after: float | None,
                error: str = "") -> None:
        """Re-sync the mirror from what the provider actually said."""
        lim_t = headers.get("x-ratelimit-limit-tokens")
        rem_t = headers.get("x-ratelimit-remaining-tokens")
        lim_r = headers.get("x-ratelimit-limit-requests")
        rem_r = headers.get("x-ratelimit-remaining-requests")
        if lim_t and float(lim_t) * 0.95 != self.tpm.capacity:
            self.tpm.capacity = float(lim_t) * 0.95          # limit changed without notice
        if lim_r and float(lim_r) != self.rpd.capacity:
            self.rpd.capacity = float(lim_r)
        self.tpm.refill()
        self.rpd.refill()
        if rem_t is not None:
            self.tpm.level = min(self.tpm.level, float(rem_t) * 0.95)
        if rem_r is not None:
            self.rpd.level = min(self.rpd.level, float(rem_r))
        if status == "rate_limited":
            self.stats["synced_429"] += 1
            wait = retry_after or 2.0
            self.blocked_until = max(self.blocked_until, time.monotonic() + wait)
            m = re.search(r"\((RPM|TPM|RPD|TPD)\): Limit (\d+), Used (\d+)", error)
            if m:
                kind, lim, used = m.group(1), float(m.group(2)), float(m.group(3))
                if kind == "RPM" and lim != self.rpm.capacity:
                    self.rpm.capacity = lim
                if kind == "RPD":
                    self.rpd.capacity, self.rpd.level = lim, max(0.0, lim - used)
                if kind == "TPD":
                    self.tpd.refill()
                    self.tpd.capacity, self.tpd.level = lim, max(0.0, lim - used)
            self._wake.set()

    def refund(self, tokens: int, request: bool = False) -> None:
        """Give back what was reserved for a call that never reached the provider."""
        self.tpm.refill()
        self.tpm.level = min(self.tpm.capacity, self.tpm.level + tokens)
        if request:
            self.rpm.refill()
            self.rpd.refill()
            self.rpm.level = min(self.rpm.capacity, self.rpm.level + 1)
            self.rpd.level = min(self.rpd.capacity, self.rpd.level + 1)
        self.stats["refunded"] = self.stats.get("refunded", 0) + 1
        self._wake.set()

    # ------------------------------------------------------------- loop
    def _ensure_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while self._heap:
            now = time.monotonic()
            # drop waiters whose deadline already passed
            alive = []
            for w in self._heap:
                if w.deadline <= now and not w.fut.done():
                    self.stats["refused_deadline"] += 1
                    w.fut.set_result(False)
                elif not w.fut.done():
                    alive.append(w)
            heapq.heapify(alive)
            self._heap = alive
            if not self._heap:
                break
            head = self._heap[0]
            wait = max(self.rpm.wait_for(1), self.tpm.wait_for(head.tokens),
                       self.rpd.wait_for(1), self.blocked_until - now)
            if wait <= 0:
                heapq.heappop(self._heap)
                self.rpm.level -= 1
                self.tpm.level -= head.tokens
                self.rpd.level -= 1
                self.stats["granted"] += 1
                head.fut.set_result(True)
                continue
            if now + wait > head.deadline:      # cannot make it: refuse now, not later
                heapq.heappop(self._heap)
                self.stats["refused_deadline"] += 1
                head.fut.set_result(False)
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=min(wait, 0.5))
            except asyncio.TimeoutError:
                pass

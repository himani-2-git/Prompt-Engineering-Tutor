from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, max_events: int, window_seconds: float):
        self.max_events, self.window = max_events, window_seconds
        self._events: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str = "global", now: float | None = None) -> bool:
        now = time.time() if now is None else now
        q = self._events[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_events:
            return False
        q.append(now)
        return True

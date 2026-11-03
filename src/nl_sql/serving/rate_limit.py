"""In-memory sliding-window rate limits keyed by visitor."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Allow at most ``count`` requests per ``window_seconds`` for each key, per window.

    State lives in process memory, so on serverless hosts each instance counts
    separately; it stops casual abuse, while the model host's quota is the hard cap.
    """

    def __init__(
        self,
        limits: list[tuple[int, float]],
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.limits = sorted(limits, key=lambda limit: limit[1])
        self.clock = clock
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a request and return None, or the seconds to wait if it is over a limit."""
        now = self.clock()
        longest = self.limits[-1][1]
        with self._lock:
            if key not in self._hits and len(self._hits) >= self.max_keys:
                self._forget_idle(now, longest)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= longest:
                hits.popleft()
            for count, window in self.limits:
                recent = [hit for hit in hits if now - hit < window]
                if len(recent) >= count:
                    return max(1.0, window - (now - recent[0]))
            hits.append(now)
            return None

    def _forget_idle(self, now: float, longest: float) -> None:
        for key in [key for key, hits in self._hits.items() if not hits or now - hits[-1] >= longest]:
            del self._hits[key]
        if len(self._hits) >= self.max_keys:
            self._hits.clear()

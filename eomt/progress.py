"""Progress of the training and validation loops: a tqdm bar on a terminal, plain lines in a log file.

tqdm redraws its bar with carriage returns several times a second; redirected to a file over a long run they pile up
(tens of MB, hundreds of thousands of partial lines). When stderr (where tqdm writes) is not a terminal,
:func:`progress` prints one line instead every :data:`LINE_EVERY_PCT` % of the loop or every :data:`LINE_EVERY_S`
seconds, whichever comes first.
"""

from __future__ import annotations

import sys
import time

from tqdm import tqdm

#: In a log file: a line every this many percent of the loop ...
LINE_EVERY_PCT = 10
#: ... or every this many seconds, whichever comes first.
LINE_EVERY_S = 300.0


def progress(iterable, *, desc: str = "", unit: str = "it", total: int | None = None, disable: bool = False,
             leave: bool = True):
    """``tqdm(iterable, ...)`` on a terminal, else a :class:`LineProgress` over it."""
    if disable or sys.stderr.isatty():
        return tqdm(iterable, desc=desc, unit=unit, total=total, disable=disable, leave=leave)
    return LineProgress(iterable, desc=desc, unit=unit, total=total)


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class LineProgress:
    """Iterates over ``iterable`` like a tqdm bar (with ``set_postfix``) and prints a plain line to stdout every
    :data:`LINE_EVERY_PCT` % or :data:`LINE_EVERY_S` seconds."""

    def __init__(self, iterable, *, desc: str = "", unit: str = "it", total: int | None = None):
        self.iterable, self.desc, self.unit = iterable, desc, unit
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.postfix = ""

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        items = {**(ordered_dict or {}), **kwargs}
        self.postfix = ", ".join(f"{k}={v}" for k, v in items.items())

    def __len__(self) -> int:
        return len(self.iterable)

    def __iter__(self):
        t0 = last = time.monotonic()
        next_pct = LINE_EVERY_PCT
        for n, item in enumerate(self.iterable, start=1):
            yield item
            now = time.monotonic()
            pct = 100.0 * n / self.total if self.total else None
            if (pct is not None and pct >= next_pct) or now - last >= LINE_EVERY_S:
                self._line(n, pct, now - t0)
                last = now
                if pct is not None:
                    next_pct = (int(pct // LINE_EVERY_PCT) + 1) * LINE_EVERY_PCT

    def _line(self, n: int, pct: float | None, elapsed: float) -> None:
        done = f"{n}/{self.total} ({pct:.0f}%)" if pct is not None else f"{n} {self.unit}"
        eta = f", {_clock(elapsed * (self.total - n) / n)} left" if pct is not None and n < self.total else ""
        rate = f", {n / elapsed:.2f} {self.unit}/s" if elapsed > 0 else ""
        post = f", {self.postfix}" if self.postfix else ""
        print(f"{self.desc}: {done}, {_clock(elapsed)} elapsed{eta}{rate}{post}", flush=True)

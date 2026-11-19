"""The ``eomt`` logger: what training, validation, prediction and loading report, one line per message on stdout.

It is a standard :mod:`logging` logger, at INFO and with its own handler (so a script sees the lines without
configuring logging). ``logging.getLogger("eomt").setLevel(logging.WARNING)`` silences it; remove its handler and set
``propagate = True`` to route the lines through your own logging setup instead.
"""

from __future__ import annotations

import logging
import sys


class _Stdout(logging.StreamHandler):
    """A handler on whatever ``sys.stdout`` is when a line is written (a redirect, pytest's capture, the null device
    the non-main processes of a multi-GPU run write to), not the one at import."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:  # StreamHandler.__init__ assigns it
        pass


LOG = logging.getLogger("eomt")
if not LOG.handlers:
    _handler = _Stdout()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(_handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

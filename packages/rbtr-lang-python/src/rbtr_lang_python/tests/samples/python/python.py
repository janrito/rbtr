"""Greeter — formats greetings for named recipients.

A sample module exercising the constructs the python plugin extracts:
functions (sync, async, decorated), classes, methods (instance, property,
static, class), module-level variables (including tuple unpacking and
annotated assignments), nested functions (scoped to their parent), PEP 695
`type` aliases (as classes), and the import styles that carry distinct
metadata.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path as P

from .config import LOCALE

type GreetingList = list[str]

DEFAULT_GREETING = "Hello"
LOCALES, FALLBACK = ("en", "fr"), "en"
MAX_RECIPIENTS: int = 100


def format_greeting(name: str) -> str:
    """Return a greeting for ``name`` in the configured locale."""

    def normalise(raw: str) -> str:
        """Trim and title-case a raw recipient name."""
        return raw.strip().title()

    return f"{DEFAULT_GREETING}, {normalise(name)} ({LOCALE})"


async def fetch_remote_greeting(url: str) -> str:
    """Fetch a greeting template from a remote source."""
    return os.environ.get("GREETING", DEFAULT_GREETING)


@lru_cache
def cached_default() -> str:
    """Cache and return the default greeting prefix."""
    return DEFAULT_GREETING


class Greeter:
    """Stateful greeter holding a prefix and recipient log."""

    def __init__(self, prefix: str = DEFAULT_GREETING) -> None:
        self.prefix = prefix
        self._seen: list[str] = []

    def greet(self, name: str) -> str:
        """Greet ``name`` and record the recipient."""
        self._seen.append(name)
        return format_greeting(name)

    @property
    def seen(self) -> list[str]:
        """Recipients greeted so far."""
        return list(self._seen)

    @staticmethod
    def shout(message: str) -> str:
        """Upper-case a message."""
        return message.upper()

    @classmethod
    def default(cls) -> Greeter:
        """Build a greeter with the default prefix."""
        return cls(DEFAULT_GREETING)


def config_path() -> P:
    """Return the path to the greeter config file."""
    return P.home() / ".greeter"

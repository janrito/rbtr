"""Per-home invariant state.

Tiny, append-only on first write — fields here pin properties
of the index that, once chosen, must not change for the lifetime
of the home.

The motivating field is `strip_docstrings_mode`. The daemon
writes one home; chunks built with stripping look different
from chunks built without. Mixing them in the same home would
poison every search. Rather than forbid `--strip-docstrings`
under `--daemon` (which prevents the eval harness from running
both modes through the daemon) we record the chosen mode at
first index and reject mismatches on subsequent calls. The user
is told to use a distinct `--home` per mode.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class HomeState(BaseModel, frozen=True):
    """Invariants pinned at the first index of a home."""

    strip_docstrings_mode: bool


def _path(home: Path) -> Path:
    return home / "home_state.json"


def load(home: Path) -> HomeState | None:
    """Return the persisted state for *home*, or None if unset."""
    p = _path(home)
    if not p.exists():
        return None
    return HomeState.model_validate_json(p.read_text(encoding="utf-8"))


def save(home: Path, state: HomeState) -> None:
    """Persist *state* under *home*. Atomic write via rename."""
    home.mkdir(parents=True, exist_ok=True)
    p = _path(home)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json() + "\n", encoding="utf-8")
    tmp.replace(p)


class StripModeMismatch(Exception):
    """Raised when an index call's strip mode disagrees with the home's pinned mode."""

    def __init__(self, *, requested: bool, pinned: bool) -> None:
        super().__init__(
            f"home was first indexed with strip_docstrings={pinned}; "
            f"refusing index with strip_docstrings={requested}. "
            "Use a distinct --home for each mode."
        )
        self.requested = requested
        self.pinned = pinned


def ensure_or_pin(home: Path, requested: bool) -> None:
    """Pin the mode if unset; reject if it disagrees with the existing pin.

    Called by every index entry-point before it builds.  After
    the call returns successfully, the home is guaranteed to be
    in *requested* mode.
    """
    existing = load(home)
    if existing is None:
        save(home, HomeState(strip_docstrings_mode=requested))
        return
    if existing.strip_docstrings_mode != requested:
        raise StripModeMismatch(
            requested=requested,
            pinned=existing.strip_docstrings_mode,
        )

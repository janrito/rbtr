"""xterm `modifyOtherKeys` preprocessing.

Translates `modifyOtherKeys` sequences into legacy bytes that
prompt_toolkit's `Vt100Parser` understands.  Called on raw stdin
data before feeding to the parser.

Pure functions — no I/O, no state.
"""

from __future__ import annotations

import re
from enum import IntEnum, IntFlag


class Modifier(IntFlag):
    """Modifier bits (1-indexed in the protocol; subtract 1)."""

    SHIFT = 1
    ALT = 2
    CTRL = 4


class Codepoint(IntEnum):
    """Well-known key codepoints."""

    TAB = 9
    ENTER = 13
    ESCAPE = 27
    SPACE = 32
    BACKSPACE = 127


# ── ASCII ranges ─────────────────────────────────────────────────────

_ORD_A = ord("a")
_ORD_Z = ord("z")
_PRINTABLE_START = ord(" ")  # 32
_PRINTABLE_END = ord("~")  # 126
_CTRL_MASK = 0x1F

# ── Legacy byte sequences ───────────────────────────────────────────

_ESC = "\x1b"
_ESC_CR = "\x1b\r"  # Escape + CR — triggers Alt+Enter handler → newline.
_BACKTAB = "\x1b[Z"
_ALT_BACKSPACE = "\x1b\x7f"
_DEL = "\x7f"
_CR = "\r"
_TAB = "\t"

# ── modifyOtherKeys pattern ─────────────────────────────────────────

# Format: \x1b[27;<modifier>;<codepoint>~
_MOK_RE = re.compile(r"\x1b\[27;(\d+);(\d+)~")

# Modifier values are 1-indexed; 1 means "no modifier".
_MOD_OFFSET = 1


def _translate_modified_key(codepoint: int, modifier: int) -> str:
    """Map a (codepoint, modifier bitmask) to legacy terminal bytes."""
    match codepoint:
        case Codepoint.ESCAPE:
            return _ESC

        case Codepoint.ENTER:
            if modifier & (Modifier.SHIFT | Modifier.ALT):
                return _ESC_CR
            return _CR

        case Codepoint.TAB:
            if modifier & Modifier.SHIFT:
                return _BACKTAB
            return _TAB

        case Codepoint.BACKSPACE:
            if modifier & Modifier.ALT:
                return _ALT_BACKSPACE
            return _DEL

        case Codepoint.SPACE:
            return " "

        case cp if _ORD_A <= cp <= _ORD_Z:
            if modifier & Modifier.CTRL:
                return chr(cp & _CTRL_MASK)
            if modifier & Modifier.ALT:
                return _ESC + chr(cp)
            return chr(cp)

        case cp if _PRINTABLE_START <= cp <= _PRINTABLE_END:
            if modifier & Modifier.ALT:
                return _ESC + chr(cp)
            return chr(cp)

        case _:  # Unknown codepoint — strip to prevent garbage.
            return ""


def _translate_mok(match: re.Match[str]) -> str:
    """Translate a single `modifyOtherKeys` sequence to legacy bytes."""
    mod_raw = int(match.group(1))
    codepoint = int(match.group(2))
    return _translate_modified_key(codepoint, mod_raw - _MOD_OFFSET)


def preprocess(data: str) -> str:
    """Translate `modifyOtherKeys` sequences to legacy bytes."""
    return _MOK_RE.sub(_translate_mok, data)

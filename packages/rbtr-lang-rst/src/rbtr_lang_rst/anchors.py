"""The anchors docutils gives a reStructuredText document's sections.

A link's fragment, and a `:ref:` role, name a section by an id: docutils
makes one from the section's title, and from each label set before it
(`.. _intro:`). `make_id` and its tables are docutils' own
(`docutils.nodes.make_id`, docutils 0.23), so the ids match what Sphinx and
docutils render.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_NON_ID_CHARS = re.compile("[^a-z0-9]+")
_NON_ID_AT_ENDS = re.compile("^[-0-9]+|-+$")
_NON_ID_TRANSLATE: dict[int, str] = {
    0x00F8: "o",  # o with stroke
    0x0111: "d",  # d with stroke
    0x0127: "h",  # h with stroke
    0x0131: "i",  # dotless i
    0x0142: "l",  # l with stroke
    0x0167: "t",  # t with stroke
    0x0180: "b",  # b with stroke
    0x0183: "b",  # b with topbar
    0x0188: "c",  # c with hook
    0x018C: "d",  # d with topbar
    0x0192: "f",  # f with hook
    0x0199: "k",  # k with hook
    0x019A: "l",  # l with bar
    0x019E: "n",  # n with long right leg
    0x01A5: "p",  # p with hook
    0x01AB: "t",  # t with palatal hook
    0x01AD: "t",  # t with hook
    0x01B4: "y",  # y with hook
    0x01B6: "z",  # z with stroke
    0x01E5: "g",  # g with stroke
    0x0225: "z",  # z with hook
    0x0234: "l",  # l with curl
    0x0235: "n",  # n with curl
    0x0236: "t",  # t with curl
    0x0237: "j",  # dotless j
    0x023C: "c",  # c with stroke
    0x023F: "s",  # s with swash tail
    0x0240: "z",  # z with swash tail
    0x0247: "e",  # e with stroke
    0x0249: "j",  # j with stroke
    0x024B: "q",  # q with hook tail
    0x024D: "r",  # r with stroke
    0x024F: "y",  # y with stroke
}
_NON_ID_TRANSLATE_DIGRAPHS: dict[int, str] = {
    0x00DF: "sz",  # ligature sz
    0x00E6: "ae",  # ae
    0x0153: "oe",  # ligature oe
    0x0238: "db",  # db digraph
    0x0239: "qp",  # qp digraph
}


def make_id(text: str) -> str:
    """The id docutils makes of *text*: ASCII letters and digits, joined by `-`, starting with a letter."""
    folded = text.lower().translate(_NON_ID_TRANSLATE_DIGRAPHS).translate(_NON_ID_TRANSLATE)
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    return _NON_ID_AT_ENDS.sub("", _NON_ID_CHARS.sub("-", " ".join(folded.split())))


def section_ids(titles: Sequence[str]) -> list[str]:
    """The id of each section of one document, given its titles in order.

    A title whose id is taken gets `-1`, `-2`, counted per id; one with no
    id at all gets `section-1`, `section-2`.
    """
    taken: set[str] = set()
    counters: Counter[str] = Counter()
    ids: list[str] = []
    for title in titles:
        base = make_id(title)
        section_id = base
        if not base or base in taken:
            prefix = f"{base}-" if base else "section-"
            while True:
                counters[prefix] += 1
                section_id = f"{prefix}{counters[prefix]}"
                if section_id not in taken:
                    break
        taken.add(section_id)
        ids.append(section_id)
    return ids

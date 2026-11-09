"""The anchors common Markdown renderers give a document's headings.

A link's fragment names a heading by its anchor, which a renderer derives
from the heading's text, each by its own rule. A section answers to the
anchor every common renderer gives it, and to one its author sets with
`{#id}`:

- GitHub's (`github-slugger`), which GitLab, Hugo and kramdown's GFM parser
  share;
- Python-Markdown's `toc` (MkDocs);
- Pandoc's `auto_identifiers`;
- kramdown's own.

Each renderer numbers a repeated anchor in its own way, so the anchors of
a document's headings are computed together, in order.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_EXPLICIT_ID = re.compile(r"\s*\{([^{}]*)\}\s*$")
_PYTHON_MARKDOWN_COUNT = re.compile(r"^(.*)_([0-9]+)$")


def _github(text: str) -> str:
    """Lowercase; drop all but letters, marks, digits, `_`, `-` and spaces; each space a `-`."""
    kept = (
        c
        for c in text.lower()
        if c in " -"
        or unicodedata.category(c)[0] in "LM"
        or unicodedata.category(c) in ("Nd", "Nl", "Pc")
    )
    return "".join(kept).replace(" ", "-")


def _python_markdown(text: str) -> str:
    """Fold to ASCII; drop all but word characters, spaces and `-`; collapse runs into one `-`."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", folded).strip().lower())


def _pandoc(text: str) -> str:
    """Keep letters, digits, `_`, `-`, `.`; join words with `-`; start at the first letter.

    Pandoc reads `--`, `---` and `...` as dashes and an ellipsis, which
    the rule then drops.
    """
    smart = re.sub(r"-{2,3}|\.\.\.", "", text).lower()
    joined = "-".join("".join(c for c in smart if c.isspace() or c.isalnum() or c in "_-.").split())
    first_letter = next((i for i, c in enumerate(joined) if c.isalpha()), len(joined))
    return joined[first_letter:] or "section"


def _kramdown(text: str) -> str:
    """Start at the first ASCII letter; keep `[a-zA-Z0-9 -]`; each space a `-`; lowercase."""
    trimmed = re.sub(r"[^a-zA-Z0-9 -]", "", re.sub(r"^[^a-zA-Z]+", "", text))
    return trimmed.replace(" ", "-").lower() or "section"


def heading_anchors(texts: Sequence[str]) -> list[tuple[str, ...]]:
    """The anchors of each heading of one document, given its rendered texts in order.

    A heading ending `{#id ...}` keeps `id` as its anchor in every renderer
    that reads the syntax; GitHub's reads it as text.
    """
    github_seen: dict[str, int] = {}
    python_markdown_seen: set[str] = set()
    pandoc_seen: set[str] = set()
    kramdown_seen: dict[str, int] = {}
    anchors: list[tuple[str, ...]] = []
    for text in texts:
        github = _github(text.strip())
        base = github
        while github in github_seen:
            github_seen[base] += 1
            github = f"{base}-{github_seen[base]}"
        github_seen[github] = 0

        attributes = _EXPLICIT_ID.search(text)
        explicit = next(
            (
                token[1:]
                for token in (attributes.group(1).split() if attributes else [])
                if token.startswith("#")
            ),
            None,
        )
        plain = text[: attributes.start()] if attributes and explicit else text

        python_markdown = explicit or _python_markdown(plain)
        if not explicit and (python_markdown in python_markdown_seen or not python_markdown):
            counted = _PYTHON_MARKDOWN_COUNT.match(python_markdown)
            stem, count = (
                (counted.group(1), int(counted.group(2))) if counted else (python_markdown, 0)
            )
            count += 1
            while f"{stem}_{count}" in python_markdown_seen:
                count += 1
            python_markdown = f"{stem}_{count}"
        python_markdown_seen.add(python_markdown)

        pandoc = explicit or _pandoc(plain)
        if not explicit and pandoc in pandoc_seen:
            pandoc = next(
                f"{pandoc}-{n}"
                for n in range(1, len(pandoc_seen) + 2)
                if f"{pandoc}-{n}" not in pandoc_seen
            )
        pandoc_seen.add(pandoc)

        kramdown = explicit or _kramdown(plain)
        if not explicit:
            if kramdown in kramdown_seen:
                kramdown_seen[kramdown] += 1
                kramdown = f"{kramdown}-{kramdown_seen[kramdown]}"
            else:
                kramdown_seen[kramdown] = 0

        anchors.append(tuple(dict.fromkeys((github, python_markdown, pandoc, kramdown))))
    return anchors

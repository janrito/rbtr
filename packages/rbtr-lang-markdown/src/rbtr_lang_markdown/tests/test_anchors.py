"""A section's anchors: what each common renderer makes of its heading, and any set explicitly.

The expected anchors are what the renderers themselves produced for the
same document: `github-slugger` 2.0.0 (GitHub; GitLab, Hugo and kramdown's
GFM parser agree), Python-Markdown 3.11.3 (MkDocs) and Pandoc 3.13.
kramdown's own rule, from its source, adds the ASCII-only anchors.
"""

from __future__ import annotations

from rbtr.domain.models import Section
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


def test_a_section_answers_to_every_renderers_anchor() -> None:
    source = """\
## Getting Started

## Hello, World!

## API v2.0 — Overview

## snake_case_name

## Multiple   spaces here

## Already-hyphenated -- twice

## Café au lait

## 日本語の見出し

## Emoji 🎉 party

## `code` in heading

## [Link text](https://example.com) heading

## C++ & C#

## 1. Numbered heading

## Duplicate

## Duplicate

## Über Größe
"""
    chunks = extract_file(FileEntry("doc.md", "sha1", source.encode()), "markdown")
    anchors = [set(c.role.meta.anchors) for c in chunks if isinstance(c.role, Section)]
    assert anchors == [
        {"getting-started"},
        {"hello-world"},
        {"api-v20--overview", "api-v20-overview", "api-v2.0-overview"},
        {"snake_case_name", "snakecasename"},
        {"multiple---spaces-here", "multiple-spaces-here"},
        {"already-hyphenated----twice", "already-hyphenated-twice"},
        {"café-au-lait", "cafe-au-lait", "caf-au-lait"},
        {"日本語の見出し", "_1", "section"},
        {"emoji--party", "emoji-party"},
        {"code-in-heading"},
        {"link-text-heading"},
        {"c--c", "c-c"},
        {"1-numbered-heading", "numbered-heading"},
        {"duplicate"},
        {"duplicate-1", "duplicate_1"},
        {"über-größe", "uber-groe", "ber-gre"},
    ]


def test_a_section_answers_to_anchors_set_in_it() -> None:
    """`{#id}` after a heading, and an `<a id>` or `<a name>` in the section's own text.

    GitHub does not read `{#id}`: its anchor slugs the braces with the
    rest of the heading. An `<a id>` in a code block is code, not an
    anchor.
    """
    source = """\
## Custom heading {#my-anchor}

Body <a id="in-para"></a> text.

<a name="own-line"></a>

```html
<a id="in-code"></a>
```

## Next
"""
    chunks = extract_file(FileEntry("doc.md", "sha1", source.encode()), "markdown")
    anchors = [set(c.role.meta.anchors) for c in chunks if isinstance(c.role, Section)]
    assert anchors == [
        {"custom-heading-my-anchor", "my-anchor", "in-para", "own-line"},
        {"next"},
    ]

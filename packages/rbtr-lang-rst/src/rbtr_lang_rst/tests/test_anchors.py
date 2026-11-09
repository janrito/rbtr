"""A section's anchors: the id docutils gives its title, and the labels set before it.

The expected ids are what docutils 0.23 itself gave the same document.
"""

from __future__ import annotations

from rbtr.domain.models import Section
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


def test_a_section_answers_to_its_docutils_id_and_its_labels() -> None:
    source = """\
Getting Started
===================

Body.

.. _greeting-label:

Hello, World!
=================

Body.

API v2.0 — Overview
=========================

Body.

snake_case_name
===================

Body.

Multiple   spaces here
==========================

Body.

Already-hyphenated -- twice
===============================

Body.

Café au lait
=================

Body.

日本語の見出し
==========================

Body.

Emoji 🎉 party
====================

Body.

C++ & C#
============

Body.

1. Numbered heading
=======================

Body.

Duplicate
=============

Body.

Duplicate
=============

Body.

Über Größe
================

Body.
"""
    chunks = extract_file(FileEntry("doc.rst", "sha1", source.encode()), "rst")
    anchors = [set(c.role.meta.anchors) for c in chunks if isinstance(c.role, Section)]
    assert anchors == [
        {"getting-started"},
        {"hello-world", "greeting-label"},
        {"api-v2-0-overview"},
        {"snake-case-name"},
        {"multiple-spaces-here"},
        {"already-hyphenated-twice"},
        {"cafe-au-lait"},
        {"section-1"},
        {"emoji-party"},
        {"c-c"},
        {"numbered-heading"},
        {"duplicate"},
        {"duplicate-1"},
        {"uber-grosze"},
    ]

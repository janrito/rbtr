"""reStructuredText language plugin: a tree-sitter query plugin (`rst.scm`).

Sections are levelled by the order each adornment style first appears,
as docutils levels them; each holds its text up to the next section. A
label is a section of its own name. Roles, embedded-target hyperlinks and
`toctree` entries are imports.

Extracted chunks::

    Guide                           → doc_section "Guide", scope ""
    =====

    .. _setup:                      → doc_section "setup", scope ""

    Setup                           → doc_section "Setup", scope "Guide"
    -----
    See :func:`~pkg.greet`.         → import {names: greet, language_hint: python}
    See :ref:`the guide <intro>`.   → import {names: intro, language_hint: rst}
    See `the API <api.rst#run>`_.   → import {module: api.rst, names: run}

    .. toctree::

       topics/intro                 → import {module: topics/intro}
       Reference <api/index>        → import {module: api/index}
"""

from __future__ import annotations

import re

from rbtr.domain.models import SectionNames
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

rst = LanguageRegistration(
    id="rst",
    extensions=frozenset({".rst"}),
    grammar_module="tree_sitter_rst",
    extraction=QueryExtraction(query=load_query(__package__, "rst")),
    prose=True,
    # Title underlines, directives and roles.
    content_signals=(
        re.compile(r"^[=\-~^\"'+#*]{3,}$", re.MULTILINE),
        re.compile(r"^\.\.\s+\w+::", re.MULTILINE),
        re.compile(r":\w+:`[^`]+`"),
    ),
    section_names=SectionNames.FOLDED,
    extraction_serial=7,
)

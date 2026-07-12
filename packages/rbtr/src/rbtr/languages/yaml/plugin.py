"""YAML language plugin.

Extracts each top-level mapping key as a doc section, via a
tree-sitter query. Nested keys are part of their parent's
content, not separate chunks.

Extracted chunks::

    name: CI                        → doc_section "name", scope ""
    on: [push]                      → doc_section "on", scope ""
    jobs:                           → doc_section "jobs", scope ""
      test:
        runs-on: ubuntu-latest
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

# Top-level keys only: pairs of the document's own block mapping.
# Nested pairs sit in a value's block mapping and do not match.


yaml = LanguageRegistration(
    id="yaml",
    extensions=frozenset({".yaml", ".yml"}),
    grammar_module="tree_sitter_yaml",
    extraction=QueryExtraction(
        query=load_query(__package__, "yaml"),
    ),
    language_plugin_version=2,
)

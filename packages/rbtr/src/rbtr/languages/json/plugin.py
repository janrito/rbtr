"""JSON language plugin.

Splits JSON by top-level keys in the root object via a
tree-sitter query. Non-object JSON (arrays, scalars) produces
no structural chunks and falls through to plaintext.

Extracted chunks::

    {                               → doc_section "name", scope ""
      "name": "my-project",         → doc_section "version", scope ""
      "version": "1.0.0",           → doc_section "dependencies", scope ""
      "dependencies": { ... }
    }
"""

from __future__ import annotations

from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

json = LanguageRegistration(
    id="json",
    extensions=frozenset({".json"}),
    grammar_module="tree_sitter_json",
    query=load_query(__package__, "json"),
)

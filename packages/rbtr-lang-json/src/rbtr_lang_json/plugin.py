"""JSON language plugin.

Splits JSON by top-level keys in the root object via a
tree-sitter query, one config-key chunk per key. Non-object JSON
(arrays, scalars) produces no structural chunks and falls through
to plaintext.

Extracted chunks::

    {                               → config_key "name", scope ""
      "name": "my-project",         → config_key "version", scope ""
      "version": "1.0.0",           → config_key "dependencies", scope ""
      "dependencies": { ... }
    }
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

json = LanguageRegistration(
    id="json",
    extensions=frozenset({".json"}),
    grammar_module="tree_sitter_json",
    extraction=QueryExtraction(
        query=load_query(__package__, "json"),
    ),
    extraction_serial=2,
)

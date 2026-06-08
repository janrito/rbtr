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

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

_QUERY = """\
(pair
  key: (string (string_content) @_section_name)) @doc_section
"""


class JsonPlugin:
    """JSON language support — top-level key extraction."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="json",
                extensions=frozenset({".json"}),
                grammar_module="tree_sitter_json",
                query=_QUERY,
            ),
        ]

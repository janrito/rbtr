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

from typing import TYPE_CHECKING

from rbtr.domain.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node

json = LanguageRegistration(
    id="json",
    extensions=frozenset({".json"}),
    grammar_module="tree_sitter_json",
    extraction=QueryExtraction(
        query=load_query(__package__, "json"),
    ),
    extraction_serial=3,
)


@json.import_extractor
def split_pointer_fragment(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
    """Separate a JSON pointer from the document it points into.

    `{"$ref": "defs.json#/Named"}` reaches `defs.json`, and the pointer
    after the `#` names what it wants inside it, which is the same shape a
    Markdown link to a heading has.
    """
    meta = resolver(node, captures)
    if "#" in meta.module:
        module, _, pointer = meta.module.partition("#")
        meta.module = module
        meta.names = pointer.lstrip("/").replace("/", ".")
    return meta

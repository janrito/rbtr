"""Python language plugin.

Provides full support: symbol extraction (functions, classes,
methods, imports) and structured import metadata.

Extracted chunks::

    def hello():            → function "hello", scope ""
        pass

    class Config:           → class "Config", scope ""
        def load(self):     → method "load", scope "Config"
            pass

    import os               → import, metadata {module: "os"}
    from pathlib import Path
                            → import, metadata {module: "pathlib", names: "Path"}
    from ..core import engine
                            → import, metadata {module: "../core", names: "engine"}

"""

from __future__ import annotations

from rbtr.domain.edge_frames import ModuleStyle
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

# The `@_docstring` sub-capture marks the first-statement body
# docstring for doc-range detection (used by `docstrings`
# and the eval query sampler).  It is optional in the match
# (`?`), so functions/classes without a docstring still match.


python = LanguageRegistration(
    id="python",
    extensions=frozenset({".py", ".pyi"}),
    grammar_module="tree_sitter_python",
    extraction=QueryExtraction(
        query=load_query(__package__, "python"),
        scope_types=frozenset({"class_definition", "function_definition"}),
        class_scope_types=frozenset({"class_definition"}),
    ),
    index_files=frozenset({"__init__.py"}),
    source_roots=("", "src"),
    module_style=ModuleStyle.DOTTED,
    extraction_serial=10,
)

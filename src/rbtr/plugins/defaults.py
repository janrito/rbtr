"""Default language registrations.

Registers languages that either:

- Have a tree-sitter grammar available but no symbol extraction
  query or import extractor yet.  Files in these languages get
  tree-sitter parsing (for potential future use) but fall back to
  line-based chunking and text-search import resolution.

- Are detection-only (Markdown, RST) and use heading-based chunking
  instead of tree-sitter.

These registrations are loaded at the lowest priority — any
specific language plugin (built-in or external) will override them.

To promote a language to full support, create a dedicated plugin
file (e.g. ``plugins/ruby.py``) with a query and optionally an
import extractor, and remove its entry here.

Example of promoting Ruby::

    # plugins/ruby.py
    from rbtr.plugins.hookspec import LanguageRegistration, hookimpl

    class RubyPlugin:
        @hookimpl
        def rbtr_register_languages(self):
            return [LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
                query='(method name: (identifier) @_fn_name) @function',
                scope_types=frozenset({"class", "module"}),
            )]

Then register it in ``plugin_manager.py``'s ``_register_builtins``.
"""

from __future__ import annotations

from rbtr.index.chunks import chunk_markdown
from rbtr.plugins.hookspec import LanguageRegistration, hookimpl


class DefaultsPlugin:
    """Detection-only and grammar-only language registrations.

    Each entry here provides the minimum: file detection via
    extensions and (where available) a grammar module for
    potential future use.  No queries, no import extractors,
    no custom scope types.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            # ── Languages with grammars but no queries yet ────────
            LanguageRegistration(
                id="c_sharp",
                extensions=frozenset({".cs"}),
                grammar_module="tree_sitter_c_sharp",
            ),
            LanguageRegistration(
                id="css",
                extensions=frozenset({".css"}),
                grammar_module="tree_sitter_css",
            ),
            LanguageRegistration(
                id="hcl",
                extensions=frozenset({".hcl", ".tf"}),
                grammar_module="tree_sitter_hcl",
            ),
            LanguageRegistration(
                id="html",
                extensions=frozenset({".html", ".htm"}),
                grammar_module="tree_sitter_html",
            ),
            LanguageRegistration(
                id="json",
                extensions=frozenset({".json"}),
                grammar_module="tree_sitter_json",
            ),
            LanguageRegistration(
                id="kotlin",
                extensions=frozenset({".kt", ".kts"}),
                grammar_module="tree_sitter_kotlin",
            ),
            LanguageRegistration(
                id="scala",
                extensions=frozenset({".scala", ".sc"}),
                grammar_module="tree_sitter_scala",
            ),
            LanguageRegistration(
                id="swift",
                extensions=frozenset({".swift"}),
                grammar_module="tree_sitter_swift",
            ),
            LanguageRegistration(
                id="toml",
                extensions=frozenset({".toml"}),
                grammar_module="tree_sitter_toml",
            ),
            LanguageRegistration(
                id="yaml",
                extensions=frozenset({".yaml", ".yml"}),
                grammar_module="tree_sitter_yaml",
            ),
            # ── Prose (heading-hierarchy chunking) ────────────────
            LanguageRegistration(
                id="markdown",
                extensions=frozenset({".md"}),
                chunker=chunk_markdown,
            ),
            LanguageRegistration(
                id="rst",
                extensions=frozenset({".rst"}),
                chunker=chunk_markdown,
            ),
        ]

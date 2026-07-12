"""Vue language plugin.

`.vue` single-file components have the same `<script>`/`<style>`/template
shape as Svelte, so they reuse the SFC chunker (`chunk_sfc`) and injection
query from `rbtr-lang-svelte`. Only the grammar differs: Vue's is bundled in
`tree-sitter-language-pack`, kept out of a plain Svelte install by living
behind this separate package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.languages.registration import LanguageRegistration, load_query
from rbtr_lang_svelte.plugin import chunk_sfc

if TYPE_CHECKING:
    from tree_sitter import Language


def _vue_grammar() -> Language:
    """Load the Vue grammar from the bundled language pack."""
    import tree_sitter_language_pack  # deferred: heavy bundled native grammars

    return tree_sitter_language_pack.get_language("vue")


# Reuses the SFC injection query shipped in rbtr-lang-svelte (cached, so this
# and svelte's own load read the file once).
vue = LanguageRegistration(
    id="vue",
    extensions=frozenset({".vue"}),
    grammar_factory=_vue_grammar,
    injection_query=load_query("rbtr_lang_svelte", "injections"),
    extraction_serial=1,
)

vue.chunker(chunk_sfc)

"""Bash language plugin.

Provides function extraction only — Bash has no import system,
class system, or module structure.

Extracted chunks::

    deploy() { echo deploying; }    → function "deploy", scope ""
    function setup { ... }          → function "setup", scope ""
    alias ll="ls -l"                → variable "ll", scope ""

No classes or methods are extracted.

An `alias` name parses as one `word` fused with its `=` (`ll=`), which no
query can split. The `name_extractor` strips the trailing `=`; no other
bash name ends in one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration, NameResolver

if TYPE_CHECKING:
    from tree_sitter import Node


# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


bash = LanguageRegistration(
    id="bash",
    extensions=frozenset({".sh", ".bash", ".zsh"}),
    filenames=frozenset(
        {
            "Makefile",
            "Dockerfile",
            "Bashrc",
            ".bashrc",
            ".bash_profile",
            ".zshrc",
        }
    ),
    grammar_module="tree_sitter_bash",
    query=load_query(__package__, "bash"),
    # Bash: `#` comments above a function attach.
    doc_comment_node_types=frozenset({"comment"}),
    language_plugin_version=3,
)


@bash.name_extractor
def _strip_alias_eq(
    resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
) -> str:
    """Default name, with the `=` the grammar fuses onto an alias removed."""
    return resolver(capture_name, node, captures).rstrip("=")

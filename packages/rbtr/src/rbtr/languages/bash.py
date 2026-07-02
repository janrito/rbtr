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

from rbtr.languages.hookspec import LanguageRegistration, hookimpl, resolve_name

if TYPE_CHECKING:
    from tree_sitter import Node


def _strip_alias_eq(capture_name: str, node: Node, captures: dict[str, list[Node]]) -> str:
    """Default name, with the `=` the grammar fuses onto an alias removed."""
    return resolve_name(capture_name, node, captures).rstrip("=")


# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_definition
  name: (word) @_fn_name) @function

(command
  name: (command_name
    (word) @_cmd)
  .
  (word) @_import_module
  (#eq? @_cmd "source")) @import

(command
  name: (command_name
    (word) @_cmd)
  .
  (word) @_import_module
  (#eq? @_cmd ".")) @import

(program
  (variable_assignment
    name: (variable_name) @_var_name) @variable)

(program
  (declaration_command
    (variable_assignment
      name: (variable_name) @_var_name)) @variable)

(command
  name: (command_name (word) @_cmd)
  argument: (concatenation (word) @_var_name)
  (#eq? @_cmd "alias")
  (#match? @_var_name "=")) @variable
"""

# ── Plugin ───────────────────────────────────────────────────────────


class BashPlugin:
    """Bash / shell language support.

    Provides function extraction only.  `scope_types` is empty
    because Bash has no class-like scoping constructs — all
    functions are top-level.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
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
                query=_QUERY,
                # Bash: `#` comments above a function attach.
                doc_comment_node_types=frozenset({"comment"}),
                name_extractor=_strip_alias_eq,
                language_plugin_version=3,
            ),
        ]

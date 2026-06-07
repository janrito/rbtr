"""Bash language plugin.

Provides function extraction only — Bash has no import system,
class system, or module structure.

Extracted chunks::

    deploy() { echo deploying; }    → function "deploy", scope ""
    function setup { ... }          → function "setup", scope ""

No imports, classes, or methods are extracted.
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

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
            ),
        ]

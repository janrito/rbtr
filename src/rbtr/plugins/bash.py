"""Bash language plugin.

Provides function extraction only — Bash has no import system,
class system, or module structure.

Extracted symbol examples::

    "my_func() { echo hello; }"  → function "my_func"
    "function deploy { ... }"    → function "deploy"

Matched file extensions: ``.sh``, ``.bash``, ``.zsh``.
Matched filenames: ``Makefile``, ``Dockerfile``, ``.bashrc``,
``.bash_profile``, ``.zshrc``, ``Bashrc``.
"""

from __future__ import annotations

from rbtr.plugins.hookspec import LanguageRegistration, hookimpl

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_definition
  name: (word) @_fn_name) @function
"""

# ── Plugin ───────────────────────────────────────────────────────────


class BashPlugin:
    """Bash / shell language support.

    Provides function extraction only.  ``scope_types`` is empty
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
                scope_types=frozenset(),
            ),
        ]

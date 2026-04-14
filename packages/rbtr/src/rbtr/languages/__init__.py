"""Language plugin manager.

Collects `LanguageRegistration` instances from all plugins,
builds lookup tables, and provides the public API for language
detection, grammar loading, and import metadata extraction.

The manager is a singleton accessed via `get_manager`.
Built-in plugins are registered in precedence order (defaults
first, then specific languages), and external plugins discovered
via the `rbtr.languages` entry-point group override both.

Example usage::

    from rbtr.languages.manager import get_manager

    mgr = get_manager()

    # Detect language from a file path.
    lang_id = mgr.detect_language("src/app.py")  # "python"

    # Load the tree-sitter grammar.
    grammar = mgr.load_grammar("python")

    # Get the registration for inspection.
    reg = mgr.get_registration("python")
    assert reg is not None
    assert reg.query is not None
    assert reg.import_extractor is not None

    # One-step detect + load.
    result = mgr.get_language("src/app.py")
    # result is ("python", <Language ...>) or None
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pluggy

from rbtr.languages.hookspec import PROJECT_NAME, LanguageHookspec, LanguageRegistration

if TYPE_CHECKING:
    from tree_sitter import Language, Node

    from rbtr.index.models import ImportMeta


class LanguageManager:
    """Central registry for language plugins.

    Use `get_manager` to obtain the cached singleton instead
    of constructing directly.

    Precedence order (later overrides earlier):

    1. `DefaultsPlugin` — detection-only and grammar-only languages.
    2. Built-in specific plugins (Python, JS/TS, Go, Rust, Java, Bash).
    3. External plugins via `rbtr.languages` entry points.

    Example — checking what's registered::

        mgr = get_manager()
        for lang_id in mgr.all_language_ids():
            reg = mgr.get_registration(lang_id)
            has_query = reg.query is not None if reg else False
            print(f"{lang_id}: query={has_query}")
    """

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT_NAME)
        self._pm.add_hookspecs(LanguageHookspec)

        # Register built-in plugins: defaults first (lowest priority),
        # then specific language plugins (override defaults).
        self._register_builtins()

        # External plugins via entry points (highest priority).
        self._pm.load_setuptools_entrypoints("rbtr.languages")

        # Build lookup tables.
        self._registrations: dict[str, LanguageRegistration] = {}
        self._ext_map: dict[str, str] = {}
        self._filename_map: dict[str, str] = {}
        self._grammar_cache: dict[str, Language | None] = {}
        self._collect()

    def _register_builtins(self) -> None:
        """Register built-in plugins in precedence order."""
        # Lazy imports to avoid circular dependencies and to keep
        # grammar packages optional.
        from rbtr.languages.bash import BashPlugin
        from rbtr.languages.c import CPlugin
        from rbtr.languages.cpp import CppPlugin
        from rbtr.languages.defaults import DefaultsPlugin
        from rbtr.languages.go import GoPlugin
        from rbtr.languages.java import JavaPlugin
        from rbtr.languages.javascript import JavaScriptPlugin
        from rbtr.languages.python import PythonPlugin
        from rbtr.languages.ruby import RubyPlugin
        from rbtr.languages.rust import RustPlugin

        # Defaults first — specific plugins override.
        self._pm.register(DefaultsPlugin())
        for plugin_cls in (
            BashPlugin,
            CPlugin,
            CppPlugin,
            GoPlugin,
            JavaPlugin,
            JavaScriptPlugin,
            PythonPlugin,
            RubyPlugin,
            RustPlugin,
        ):
            self._pm.register(plugin_cls())

    def _collect(self) -> None:
        """Collect registrations from all plugins and build maps.

        Later registrations override earlier ones (specific plugins
        override defaults, external plugins override built-ins).

        Raises `ValueError` if a single plugin returns duplicate IDs.
        """
        results: list[list[LanguageRegistration]] = self._pm.hook.rbtr_register_languages()
        for batch in results:
            seen: set[str] = set()
            for reg in batch:
                if reg.id in seen:
                    msg = f"plugin returned duplicate language id {reg.id!r}"
                    raise ValueError(msg)
                seen.add(reg.id)
                self._registrations[reg.id] = reg
                for ext in reg.extensions:
                    self._ext_map[ext] = reg.id
                for name in reg.filenames:
                    self._filename_map[name] = reg.id

    # ── Public API ───────────────────────────────────────────────────

    def detect_language(self, file_path: str) -> str | None:
        """Detect the language of a file from its extension or filename.

        Tries filename first (for files like `Makefile`), then
        falls back to extension.

        Examples::

            >>> mgr = get_manager()
            >>> mgr.detect_language("src/app.py")
            'python'
            >>> mgr.detect_language("Makefile")
            'bash'
            >>> mgr.detect_language("styles.css")
            'css'
            >>> mgr.detect_language("README") is None
            True
        """
        p = PurePosixPath(file_path)
        if p.name in self._filename_map:
            return self._filename_map[p.name]
        return self._ext_map.get(p.suffix)

    def get_registration(self, language_id: str) -> LanguageRegistration | None:
        """Return the registration for *language_id*, or `None`.

        Examples::

            >>> mgr = get_manager()
            >>> reg = mgr.get_registration("python")
            >>> reg.extensions
            frozenset({'.py', '.pyi'})
            >>> mgr.get_registration("nonexistent") is None
            True
        """
        return self._registrations.get(language_id)

    def load_grammar(self, language_id: str) -> Language | None:
        """Load the tree-sitter grammar for *language_id*.

        Returns the `Language` object, or `None` if:

        - No registration exists for *language_id*.
        - The registration has no `grammar_module`.
        - The grammar package is not installed.

        Results are cached — repeated calls return the same object.

        Examples::

            >>> mgr = get_manager()
            >>> grammar = mgr.load_grammar("python")
            >>> grammar is not None
            True
            >>> mgr.load_grammar("markdown") is None  # no grammar
            True
        """
        if language_id in self._grammar_cache:
            return self._grammar_cache[language_id]

        reg = self._registrations.get(language_id)
        if reg is None or reg.grammar_module is None:
            self._grammar_cache[language_id] = None
            return None

        try:
            from tree_sitter import Language as TSLanguage  # deferred: heavy native lib

            mod = importlib.import_module(reg.grammar_module)
            lang_fn = getattr(mod, reg.grammar_entry)
            grammar = TSLanguage(lang_fn())
            self._grammar_cache[language_id] = grammar
        except (ImportError, AttributeError, OSError):
            self._grammar_cache[language_id] = None

        return self._grammar_cache[language_id]

    def get_query(self, language_id: str) -> str | None:
        """Return the tree-sitter query string for *language_id*.

        Returns `None` if no registration exists or the
        registration has no query.

        Examples::

            >>> mgr = get_manager()
            >>> q = mgr.get_query("python")
            >>> "@function" in q
            True
            >>> mgr.get_query("json") is None  # no query
            True
        """
        reg = self._registrations.get(language_id)
        return reg.query if reg else None

    def extract_import_meta(self, language_id: str, node: Node) -> ImportMeta:
        """Extract import metadata using the language's extractor.

        Calls the `import_extractor` registered for *language_id*.
        Returns an empty dict when no extractor is registered —
        `edges.py` will fall back to text search.

        Examples::

            >>> mgr = get_manager()
            >>> # With a real tree-sitter Node for "import os":
            >>> meta = mgr.extract_import_meta("python", node)
            >>> meta  # {"module": "os"}

            >>> # Language without extractor:
            >>> meta = mgr.extract_import_meta("ruby", node)
            >>> meta  # {}
        """
        reg = self._registrations.get(language_id)
        if reg and reg.import_extractor:
            return reg.import_extractor(node)
        return {}

    def get_scope_types(self, language_id: str) -> frozenset[str]:
        """Return the scope node types for *language_id*.

        Used by `extract_symbols` to detect methods inside classes.
        Returns an empty frozenset for unregistered languages.

        Examples::

            >>> mgr = get_manager()
            >>> mgr.get_scope_types("python")
            frozenset({'class_definition'})
            >>> mgr.get_scope_types("rust")
            frozenset({'impl_item', 'struct_item'})
            >>> mgr.get_scope_types("bash")
            frozenset()
        """
        reg = self._registrations.get(language_id)
        if reg:
            return reg.scope_types
        return frozenset()

    def get_language(self, file_path: str) -> tuple[str, Language] | None:
        """Detect language and load grammar in one step.

        Returns `(language_id, grammar)` or `None` if the file
        type is unrecognised or the grammar is not installed.

        Examples::

            >>> mgr = get_manager()
            >>> result = mgr.get_language("app.py")
            >>> result[0]
            'python'
            >>> mgr.get_language("data.xyz") is None
            True
        """
        lang_id = self.detect_language(file_path)
        if lang_id is None:
            return None
        grammar = self.load_grammar(lang_id)
        if grammar is None:
            return None
        return lang_id, grammar

    def missing_grammar(self, language_id: str) -> bool:
        """Check whether a language *expects* a grammar but it's not installed.

        Returns `True` only when the registration specifies a
        `grammar_module` but loading it fails.  Languages that
        intentionally have no grammar (e.g. Markdown with a custom
        chunker) return `False`.

        Examples::

            >>> mgr = get_manager()
            >>> mgr.missing_grammar("python")
            False
            >>> mgr.missing_grammar("markdown")  # custom chunker, no grammar needed
            False
        """
        reg = self._registrations.get(language_id)
        if reg is None or reg.grammar_module is None:
            return False
        return self.load_grammar(language_id) is None

    def get_pygments_lexer(self, file_path: str) -> str:
        """Return the Pygments lexer name for a file path.

        Uses `detect_language` and the registration's
        `pygments_lexer` field.  Returns `"text"` for
        unrecognised files.

        Examples::

            >>> mgr = get_manager()
            >>> mgr.get_pygments_lexer("app.py")
            'python'
            >>> mgr.get_pygments_lexer("Program.cs")
            'csharp'
            >>> mgr.get_pygments_lexer("data.xyz")
            'text'
        """
        lang_id = self.detect_language(file_path)
        if lang_id is None:
            return "text"
        reg = self._registrations[lang_id]
        return reg.pygments_lexer if reg.pygments_lexer is not None else reg.id

    def all_language_ids(self) -> list[str]:
        """Return all registered language IDs.

        Examples::

            >>> mgr = get_manager()
            >>> "python" in mgr.all_language_ids()
            True
        """
        return list(self._registrations)


@lru_cache(1)
def get_manager() -> LanguageManager:
    """Return the cached singleton `LanguageManager`.

    The manager is created on first call and reused thereafter.
    Call `reset_manager` to force re-creation (testing only).

    Example::

        from rbtr.languages.manager import get_manager

        mgr = get_manager()
        lang = mgr.detect_language("main.go")  # "go"
    """
    return LanguageManager()


def reset_manager() -> None:
    """Clear the cached singleton so the next `get_manager`
    call creates a fresh `LanguageManager`.

    Intended for tests that register custom plugins or need a
    clean slate.

    Example::

        from rbtr.languages.manager import get_manager, reset_manager

        reset_manager()
        mgr = get_manager()  # fresh instance
    """
    get_manager.cache_clear()

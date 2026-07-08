"""Language plugin manager.

Collects `LanguageRegistration` instances from all plugins,
builds lookup tables, and provides the public API for language
detection, grammar loading, and import metadata extraction.

The manager is a singleton accessed via `get_manager`.
Built-in plugins are registered in precedence order (defaults
first, then specific languages), and external plugins discovered
via the `rbtr.languages` entry-point group override both.

Example usage::

    from rbtr.languages import get_manager

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

import pluggy
from tree_sitter import Language

from rbtr.languages.bash.plugin import BashPlugin
from rbtr.languages.c.plugin import CPlugin
from rbtr.languages.cpp.plugin import CppPlugin
from rbtr.languages.css.plugin import CssPlugin
from rbtr.languages.go.plugin import GoPlugin
from rbtr.languages.hcl.plugin import HclPlugin
from rbtr.languages.hookspec import PROJECT_NAME, LanguageHookspec, LanguageRegistration
from rbtr.languages.html.plugin import HtmlPlugin
from rbtr.languages.java.plugin import JavaPlugin
from rbtr.languages.javascript.plugin import JavaScriptPlugin
from rbtr.languages.json.plugin import JsonPlugin
from rbtr.languages.less.plugin import LessPlugin
from rbtr.languages.markdown.plugin import MarkdownPlugin
from rbtr.languages.python.plugin import PythonPlugin
from rbtr.languages.rst.plugin import RstPlugin
from rbtr.languages.ruby.plugin import RubyPlugin
from rbtr.languages.rust.plugin import RustPlugin
from rbtr.languages.scss.plugin import ScssPlugin
from rbtr.languages.sfc.plugin import SveltePlugin, VuePlugin
from rbtr.languages.sql.plugin import SqlPlugin
from rbtr.languages.toml.plugin import TomlPlugin
from rbtr.languages.yaml.plugin import YamlPlugin


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
        for plugin_cls in (
            BashPlugin,
            CPlugin,
            CppPlugin,
            CssPlugin,
            GoPlugin,
            HclPlugin,
            HtmlPlugin,
            JavaPlugin,
            JavaScriptPlugin,
            JsonPlugin,
            LessPlugin,
            MarkdownPlugin,
            PythonPlugin,
            RstPlugin,
            RubyPlugin,
            RustPlugin,
            ScssPlugin,
            SveltePlugin,
            VuePlugin,
            SqlPlugin,
            TomlPlugin,
            YamlPlugin,
        ):
            self._pm.register(plugin_cls())

    def _collect(self) -> None:
        """Collect registrations from all plugins and build maps.

        Later registrations override earlier ones (specific plugins
        override defaults, external plugins override built-ins).

        Raises `ValueError` if a single plugin returns duplicate IDs.
        """
        results: list[list[LanguageRegistration]] = self._pm.hook.rbtr_register_languages()
        for batch in reversed(results):
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

        Examples:

            `detect_language("src/app.py")` → `"python"`
            `detect_language("Makefile")` → `"bash"`
            `detect_language("styles.css")` → `"css"`
            `detect_language("README")` → `None`
        """
        p = PurePosixPath(file_path)
        if p.name in self._filename_map:
            return self._filename_map[p.name]
        return self._ext_map.get(p.suffix)

    def get_registration(self, language_id: str) -> LanguageRegistration | None:
        """Return the registration for *language_id*, or `None`.

        Examples:

            `get_registration("python").extensions` → `frozenset({'.py', '.pyi'})`
            `get_registration("nonexistent")` → `None`
        """
        return self._registrations.get(language_id)

    def load_grammar(self, language_id: str) -> Language | None:
        """Load the tree-sitter grammar for *language_id*.

        Returns the `Language` object, or `None` if:

        - No registration exists for *language_id*.
        - The registration has no `grammar_module`.
        - The grammar package is not installed.

        Results are cached — repeated calls return the same object.

        Examples:

            `load_grammar("python")` → `Language` (if installed)
            `load_grammar("markdown")` → `None` (no grammar)
        """
        if language_id in self._grammar_cache:
            return self._grammar_cache[language_id]

        reg = self._registrations.get(language_id)
        if reg is None:
            self._grammar_cache[language_id] = None
            return None

        try:
            if reg.grammar_factory is not None:
                self._grammar_cache[language_id] = reg.grammar_factory()
            elif reg.grammar_module is not None:
                mod = importlib.import_module(reg.grammar_module)
                lang_fn = getattr(mod, reg.grammar_entry)
                self._grammar_cache[language_id] = Language(lang_fn())
            else:
                self._grammar_cache[language_id] = None
        except (ImportError, AttributeError, OSError):
            self._grammar_cache[language_id] = None

        return self._grammar_cache[language_id]

    def get_query(self, language_id: str) -> str | None:
        """Return the tree-sitter query string for *language_id*.

        Returns `None` if no registration exists or the
        registration has no query.

        Examples:

            `"@function" in get_query("python")` → `True`
            `get_query("json")` → `None` (no query)
        """
        reg = self._registrations.get(language_id)
        return reg.query if reg else None

    def get_scope_types(self, language_id: str) -> frozenset[str]:
        """Return the scope node types for *language_id*.

        Used by `extract_symbols` to detect methods inside classes.
        Returns an empty frozenset for unregistered languages.

        Examples:

            `get_scope_types("python")` → `frozenset({'class_definition'})`
            `get_scope_types("rust")` → `frozenset({'impl_item', 'struct_item'})`
            `get_scope_types("bash")` → `frozenset()`
        """
        reg = self._registrations.get(language_id)
        if reg:
            return reg.scope_types
        return frozenset()

    def get_language(self, file_path: str) -> tuple[str, Language] | None:
        """Detect language and load grammar in one step.

        Returns `(language_id, grammar)` or `None` if the file
        type is unrecognised or the grammar is not installed.

        Examples:

            `get_language("app.py")` → `("python", Language)`
            `get_language("data.xyz")` → `None`
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

        Examples:

            `missing_grammar("python")` → `False` (installed)
            `missing_grammar("markdown")` → `False` (custom chunker, no grammar needed)
        """
        reg = self._registrations.get(language_id)
        if reg is None or reg.grammar_module is None:
            return False
        return self.load_grammar(language_id) is None

    def all_language_ids(self) -> list[str]:
        """Return all registered language IDs.

        Examples:

            `"python" in all_language_ids()` → `True`
        """
        return list(self._registrations)


@lru_cache(1)
def get_manager() -> LanguageManager:
    """Return the cached singleton `LanguageManager`.

    The manager is created on first call and reused thereafter.
    Call `reset_manager` to force re-creation (testing only).

    Example::

        from rbtr.languages import get_manager

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

        from rbtr.languages import get_manager, reset_manager

        reset_manager()
        mgr = get_manager()  # fresh instance
    """
    get_manager.cache_clear()

"""Language plugin manager.

Collects `LanguageRegistration` instances from all plugins,
builds lookup tables, and provides the public API for language
detection, grammar loading, and import metadata extraction.

The manager is a singleton accessed via `get_manager`. All
languages are discovered from the `rbtr.languages` entry-point
group; each entry point resolves to a `LanguageRegistration`.

Example usage::

    from rbtr.languages.manager import get_manager

    mgr = get_manager()

    # Detect language from a file path.
    lang_id = mgr.detect_language("src/app.py")  # "python"

    # Load the tree-sitter grammar.
    grammar = mgr.grammar("python")

    # Get the registration for inspection.
    reg = mgr.get_registration("python")
    assert reg is not None
    assert reg.extraction is not None
"""

from __future__ import annotations

import importlib
import importlib.metadata
from functools import lru_cache
from pathlib import PurePosixPath

import structlog
from tree_sitter import Language

from rbtr.errors import RbtrError
from rbtr.languages.registration import LanguageRegistration

log = structlog.get_logger(__name__)


class LanguageManager:
    """Central registry for language plugins.

    Use `get_manager` to obtain the cached singleton instead
    of constructing directly.

    Languages are discovered from the `rbtr.languages` entry-point
    group; each entry point resolves to a `LanguageRegistration`, and
    a duplicate language id raises `RbtrError`.

    Example — checking what's registered::

        mgr = get_manager()
        for lang_id in mgr.all_language_ids():
            reg = mgr.get_registration(lang_id)
            has_extraction = reg.extraction is not None if reg else False
            print(f"{lang_id}: extraction={has_extraction}")
    """

    def __init__(self) -> None:
        self._registrations: dict[str, LanguageRegistration] = {}
        self._ext_map: dict[str, str] = {}
        self._filename_map: dict[str, str] = {}
        self._grammar_cache: dict[str, Language | None] = {}
        self._distributions: dict[str, tuple[str, str]] = {}
        self._collect()

    def _collect(self) -> None:
        """Discover registrations from the `rbtr.languages` entry-point group.

        Each entry point resolves to a `LanguageRegistration` value. A
        plugin whose module fails to import is logged and skipped, so one
        broken plugin cannot disable the rest. Two plugins claiming the
        same language id is a conflict and raises `RbtrError`.
        """
        for ep in importlib.metadata.entry_points(group="rbtr.languages"):
            try:
                reg = ep.load()
            except Exception:  # noqa: BLE001  # a broken plugin must not sink the rest
                log.warning("language_plugin_load_failed", entry_point=ep.name, exc_info=True)
                continue
            if reg.id in self._registrations:
                msg = f"duplicate language id {reg.id!r} (entry point {ep.name!r})"
                raise RbtrError(msg)
            self._registrations[reg.id] = reg
            for ext in reg.extensions:
                self._ext_map[ext] = reg.id
            for name in reg.filenames:
                self._filename_map[name] = reg.id
            dist = getattr(ep, "dist", None)
            if dist is not None:
                self._distributions[reg.id] = (dist.name, dist.version)

    # ── Public API ───────────────────────────────────────────────────

    def detect_language(self, file_path: str) -> str | None:
        """Detect the language of a file from its extension or filename.

        Tries filename first (for files like `.bashrc`), then
        falls back to extension.

        Examples:

            `detect_language("src/app.py")` → `"python"`
            `detect_language(".bashrc")` → `"bash"`
            `detect_language("styles.css")` → `"css"`
            `detect_language("README")` → `None`
        """
        p = PurePosixPath(file_path)
        if p.name in self._filename_map:
            return self._filename_map[p.name]
        return self._ext_map.get(p.suffix)

    def get_registration(self, language_id: str) -> LanguageRegistration | None:
        """Return the registration for *language_id*, or `None`."""
        return self._registrations.get(language_id)

    def grammar(self, language_id: str) -> Language | None:
        """Return the tree-sitter grammar (`Language`) for *language_id*.

        `None` if:

        - No registration exists for *language_id*.
        - The registration has no `grammar_module`.
        - The grammar package is not installed.

        Results are cached — repeated calls return the same object.

        Examples:

            `grammar("python")` → `Language` (if installed)
            `grammar("markdown")` → `None` (no grammar)
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
        return self.grammar(language_id) is None

    def all_language_ids(self) -> list[str]:
        """Return all registered language IDs."""
        return list(self._registrations)

    def distribution(self, language_id: str) -> tuple[str, str] | None:
        """Return the `(package, version)` that ships *language_id*, or `None`.

        Read from the `rbtr.languages` entry point at discovery time — the
        packaging complement to the registration (which carries
        `extraction_serial`). `None` for a registration injected without
        distribution metadata (e.g. in tests), never for a real install.
        """
        return self._distributions.get(language_id)


@lru_cache(1)
def get_manager() -> LanguageManager:
    """Return the cached singleton `LanguageManager`.

    Created on first call and reused thereafter. Call `reset_manager` to force
    re-creation (testing only).
    """
    return LanguageManager()


def reset_manager() -> None:
    """Clear the cached singleton so the next `get_manager` creates a fresh
    `LanguageManager`. For tests that register custom plugins or need a clean
    slate.
    """
    get_manager.cache_clear()

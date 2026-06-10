"""Edge inference — imports, tests, docs relationships.

All public functions are pure: chunks in, edges out.  No I/O, no
store access.  The orchestrator feeds them chunks and writes
the results.

Two resolution strategies are used:

- **Structural (tree-sitter):** Import metadata is extracted by
  `treesitter.py` during parsing and stored in `chunk.metadata`.
  This gives exact module paths and symbol names.  Used for languages
  with a tree-sitter import extractor (Python, JavaScript, TypeScript,
  Ruby, Rust).

- **Text search fallback:** When tree-sitter metadata is unavailable
  (unsupported language, doc sections, config files), we fall back to
  text matching — scanning chunk content for known symbol names or
  file-path fragments.  Less precise but works everywhere.

Doc↔code edges always use text search since docs are prose, not code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind, ImportMeta
from rbtr.languages import LanguageManager
from rbtr.languages.hookspec import ModuleStyle, parse_path_relative


@dataclass(frozen=True)
class ImportResolution:
    """File resolution hints for imports from a specific language.

    Built by the orchestrator from plugin registrations and
    passed to edge inference functions as plain data.
    """

    extensions: tuple[str, ...]
    """File extensions to try (collected from target languages)."""
    index_files: tuple[str, ...]
    """Directory entry points to try (collected from target languages)."""
    source_roots: tuple[str, ...]
    """Directories to prepend when resolving absolute imports."""
    path_substitutions: tuple[tuple[str, str], ...]
    """Module path prefix replacements (e.g. `("crate/", "src/")`)."""
    test_prefix: str = ""
    """Test file name prefix (e.g. `test_`)."""
    test_suffix: str = ""
    """Test file name suffix (e.g. `.test`, `_test`)."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How module strings map to file paths. See `ModuleStyle`."""
    own_extensions: frozenset[str] = frozenset()
    """The importing language's *own* extensions (distinct from the flattened
    target `extensions`). Used to break same-path ties in the suffix tier."""


def build_resolution_map(mgr: LanguageManager) -> dict[str, ImportResolution]:
    """Build language → resolution hints from plugin registrations."""
    result: dict[str, ImportResolution] = {}
    for lang_id in mgr.all_language_ids():
        reg = mgr.get_registration(lang_id)
        if reg is None:
            continue
        target_langs = reg.import_targets or frozenset({lang_id})
        exts: list[str] = []
        idxs: list[str] = []
        for target in target_langs:
            target_reg = mgr.get_registration(target)
            if target_reg is not None:
                exts.extend(target_reg.extensions)
                idxs.extend(target_reg.index_files)
        result[lang_id] = ImportResolution(
            extensions=tuple(exts),
            index_files=tuple(idxs),
            source_roots=reg.source_roots,
            path_substitutions=reg.path_substitutions,
            test_prefix=reg.test_prefix,
            test_suffix=reg.test_suffix,
            module_style=reg.module_style,
            own_extensions=reg.extensions,
        )
    return result


# ── Shared helpers ───────────────────────────────────────────────────


def _resolve_module_to_file(
    module_path: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str = "",
) -> str | None:
    """Find a repo file matching a `/`-separated *module_path*.

    DOTTED: converts `.` to `/` before searching.
    PATH: searches as-is (bare path first, then extensions).

    Resolution proceeds in tiers: config-driven prefix matching against
    `source_roots` first, then a layout-independent full-path *suffix* match
    (Tier 3). *importer_ext* is the importing file's extension, used only to
    break ties when several files share one path but differ by extension.
    """
    # Apply prefix substitutions.
    resolved = module_path
    for prefix, replacement in resolution.path_substitutions:
        if resolved.startswith(prefix):
            resolved = replacement + resolved[len(prefix) :]
            break

    # Style-specific conversion.
    if resolution.module_style is ModuleStyle.DOTTED and "." in resolved:
        resolved = resolved.replace(".", "/")

    # Search source_roots x (bare, extensions, index_files).
    for root in resolution.source_roots:
        base = f"{root}/{resolved}" if root else resolved
        if base in repo_files:
            return base
        for ext in resolution.extensions:
            if f"{base}{ext}" in repo_files:
                return f"{base}{ext}"
        for idx in resolution.index_files:
            if f"{base}/{idx}" in repo_files:
                return f"{base}/{idx}"

    # Tier 3: layout-independent full-path suffix match. Requires >=2 path
    # segments so single-segment bare modules (`os`, `io`) cannot match a
    # nested `.../os.py`. Files only; the index-file (package) form is not
    # tried here.
    if "/" not in resolved:
        return None
    candidates = _suffix_matches(resolved, repo_files, resolution.extensions)
    if not candidates:
        return None
    # Group by path-without-extension; more than one group is genuine
    # ambiguity (distinct files) and is dropped.
    groups: dict[str, list[str]] = {}
    for f in candidates:
        groups.setdefault(str(PurePosixPath(f).with_suffix("")), []).append(f)
    if len(groups) != 1:
        return None
    group = next(iter(groups.values()))
    if len(group) == 1:
        return group[0]
    return _pick_by_importer(group, importer_ext, resolution.own_extensions)


def _suffix_matches(
    name: str,
    repo_files: set[str],
    extensions: tuple[str, ...],
) -> list[str]:
    """Collect repo files matching *name* as a full-path suffix.

    A file matches when it equals `name{ext}` or ends with `/name{ext}` for
    some extension. Shared by the import resolver's Tier 3 and the test-edge
    `_find_source_file` last resort so both apply one suffix-matching policy.
    """
    matches: list[str] = []
    for ext in extensions:
        target = f"{name}{ext}"
        suffix = f"/{target}"
        matches.extend(f for f in repo_files if f == target or f.endswith(suffix))
    return matches


def _pick_by_importer(
    candidates: list[str],
    importer_ext: str,
    own_extensions: frozenset[str],
) -> str | None:
    """Pick the candidate closest to the importing file.

    Ranks by: (0) extension equals the importer's own extension, (1) extension
    belongs to the importer's language, (2) anything else. Returns the single
    best-ranked candidate, or `None` if two candidates tie at the top rank.
    """

    def rank(f: str) -> int:
        ext = PurePosixPath(f).suffix
        if importer_ext and ext == importer_ext:
            return 0
        if ext in own_extensions:
            return 1
        return 2

    ranked = sorted(candidates, key=rank)
    best = rank(ranked[0])
    if sum(1 for f in ranked if rank(f) == best) != 1:
        return None
    return ranked[0]


def _resolve_import_to_file(
    meta: ImportMeta,
    file_path: str,
    repo_files: set[str],
    resolution: ImportResolution | None,
) -> str | None:
    """Resolve import metadata to a repo file path.

    Handles relative imports (extractor-set `dots` or PATH-style
    `./`/`../` prefixes), then delegates to `_resolve_module_to_file`.
    """
    module = meta.module or ""

    # Determine relative depth: from extractor or from path prefix.
    dots = int(meta.dots) if meta.dots else 0
    if not dots and resolution is not None and resolution.module_style is ModuleStyle.PATH:
        dots, module = parse_path_relative(module)

    # Resolve relative path against the importing file.
    if dots:
        parts = PurePosixPath(file_path).parts
        if len(parts) < dots:
            return None
        segs = list(parts[:-dots])
        if module:
            segs.extend(s for s in module.split("/") if s) if "/" in module else segs.append(module)
        if not segs:
            return None
        module = "/".join(segs)

    if not module or resolution is None:
        return None
    importer_ext = PurePosixPath(file_path).suffix
    return _resolve_module_to_file(module, repo_files, resolution, importer_ext)


def _build_symbol_index(chunks: list[Chunk]) -> dict[tuple[str, str], Chunk]:
    """Map `(file_path, name)` → chunk for symbol lookup."""
    index: dict[tuple[str, str], Chunk] = {}
    for c in chunks:
        if c.kind in (ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD):
            key = (c.file_path, c.name)
            # First definition wins (top-level preferred over nested).
            if key not in index:
                index[key] = c
    return index


def _build_file_chunks_index(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Map `file_path` → all non-import chunks in that file.

    Used for bare/whole-file imports where the entire file is
    brought into scope.  IMPORT chunks are excluded — they
    describe the file's own dependencies, not content that
    importers depend on.
    """
    index: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.kind != ChunkKind.IMPORT:
            index.setdefault(c.file_path, []).append(c)
    return index


# ── Import edges ─────────────────────────────────────────────────────
#
# Strategy: structural (tree-sitter metadata) with text-search fallback.
#
# When tree-sitter metadata is present (`chunk.metadata` has `module`
# or `dots` keys), we use the exact module path and symbol names.
# When metadata is absent, we fall back to scanning the import chunk's
# text for known repo file stems — less precise but handles any language.
#
# Prose-language imports (markdown, RST) produce DOCUMENTS edges
# instead of IMPORTS.  Name-only imports (RST :func: etc.) search
# the symbol index by name across all files.


def _structural_import_edges(
    imp: Chunk,
    symbol_index: dict[tuple[str, str], Chunk],
    file_chunks_index: dict[str, list[Chunk]],
    repo_files: set[str],
    resolution: ImportResolution | None = None,
) -> list[Edge]:
    """Resolve import edges using tree-sitter metadata (exact).

    Named imports (`from X import Y`) link to each named symbol.
    Bare imports (`import X`, `@import "x.css"`, `#include "x.h"`)
    link to *every* non-import chunk in the target file — the
    entire file is brought into scope.
    """
    edge_kind = EdgeKind.DOCUMENTS if imp.language in _PROSE_LANGUAGES else EdgeKind.IMPORTS

    target_file = _resolve_import_to_file(imp.metadata, imp.file_path, repo_files, resolution)
    if target_file is None:
        return []

    edges: list[Edge] = []
    names_str = imp.metadata.names

    if names_str:
        # from foo import Bar, Baz → link to each named symbol.
        for name in names_str.split(","):
            target = symbol_index.get((target_file, name))
            if target is None:
                # Try DOC_SECTION name match (fragment references).
                for c in file_chunks_index.get(target_file, []):
                    if c.kind == ChunkKind.DOC_SECTION and c.name == name:
                        target = c
                        break
            if target is not None:
                edges.append(Edge(source_id=imp.id, target_id=target.id, kind=edge_kind))
    else:
        # Bare import → edge to every non-import chunk in the file.
        for target in file_chunks_index.get(target_file, []):
            edges.append(Edge(source_id=imp.id, target_id=target.id, kind=edge_kind))

    return edges


def _build_stem_index(
    repo_files: set[str],
) -> dict[str, list[str]]:
    """Map file stems → file paths, excluding `__init__` and short stems."""
    stem_to_files: dict[str, list[str]] = {}
    for f in repo_files:
        stem = PurePosixPath(f).stem
        if stem != "__init__" and len(stem) >= 3:
            stem_to_files.setdefault(stem, []).append(f)
    return stem_to_files


# Pre-compiled word tokeniser — splits on non-alphanumeric/underscore.
_WORD_RE = re.compile(r"[A-Za-z_]\w*")

# Languages whose imports produce DOCUMENTS edges instead of IMPORTS.
_PROSE_LANGUAGES = frozenset({"markdown", "rst"})


def _build_file_symbol_index(
    symbol_index: dict[tuple[str, str], Chunk],
) -> dict[str, Chunk]:
    """Map file_path → first symbol chunk for quick file-level linking."""
    result: dict[str, Chunk] = {}
    for (file_path, _name), chunk in symbol_index.items():
        if file_path not in result:
            result[file_path] = chunk
    return result


def _text_search_import_edges(
    imp: Chunk,
    file_symbols: dict[str, Chunk],
    stem_index: dict[str, list[str]],
) -> list[Edge]:
    """Resolve import edges via text search (fallback).

    Tokenises the import statement into words and looks up matching
    repo file stems via set intersection — O(words) instead of
    O(stems) regex calls.
    """
    words = set(_WORD_RE.findall(imp.name))
    edges: list[Edge] = []

    for word in words:
        files = stem_index.get(word)
        if files is None:
            continue
        for target_file in files:
            target = file_symbols.get(target_file)
            if target is not None:
                edges.append(Edge(source_id=imp.id, target_id=target.id, kind=EdgeKind.IMPORTS))

    return edges


def infer_import_edges(
    chunks: list[Chunk],
    repo_files: set[str],
    resolution_map: dict[str, ImportResolution] | None = None,
) -> list[Edge]:
    """Infer `IMPORTS` edges from import chunks.

    Uses tree-sitter metadata when available (exact), falls back to
    text search otherwise (best-effort).  When *resolution_map* is
    provided, file resolution uses language-aware extensions and
    source roots instead of the hardcoded `.py` fallback.
    """
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    symbol_index = _build_symbol_index(chunks)
    file_chunks_index = _build_file_chunks_index(chunks)
    stem_index = _build_stem_index(repo_files)
    file_symbols = _build_file_symbol_index(symbol_index)
    edges: list[Edge] = []

    for imp in imports:
        has_file = bool(imp.metadata.module or imp.metadata.dots)
        has_names_only = (
            bool(imp.metadata.names) and not has_file and imp.language in _PROSE_LANGUAGES
        )

        if has_names_only:
            # Name-only lookup (RST :func:, :class: etc.).
            # Search the symbol index by name across all files.
            edge_kind = EdgeKind.DOCUMENTS if imp.language in _PROSE_LANGUAGES else EdgeKind.IMPORTS
            for name in imp.metadata.names.split(","):
                for (_, sym_name), chunk in symbol_index.items():
                    if sym_name == name:
                        edges.append(Edge(source_id=imp.id, target_id=chunk.id, kind=edge_kind))
                        break
        elif has_file:
            # Structural: tree-sitter gave us exact module/names.
            # When language_hint is set (e.g. HTML <script src> →
            # "javascript"), resolve using the hinted language's
            # import_targets instead of the source language.
            if imp.metadata.language_hint and resolution_map:
                resolution = resolution_map.get(imp.metadata.language_hint)
            else:
                resolution = resolution_map.get(imp.language) if resolution_map else None
            edges.extend(
                _structural_import_edges(
                    imp, symbol_index, file_chunks_index, repo_files, resolution
                )
            )
        else:
            # Text search fallback: scan for repo file stems.
            edges.extend(_text_search_import_edges(imp, file_symbols, stem_index))

    return edges


# ── Test↔code edges ─────────────────────────────────────────────────
#
# Strategy: file-naming heuristic + import analysis (structural or
# text-search depending on what's available in the import chunks).


def _strip_test_affix(
    file_path: str,
    *,
    test_prefix: str = "test_",
    test_suffix: str = "",
) -> str | None:
    """Extract the base name from a test file path.

    Tries the language's prefix first, then suffix::

        test_prefix="test_": tests/test_foo.py → foo
        test_suffix=".test": src/models.test.ts → models
        test_suffix="_test": pkg/foo_test.go → foo
        test_suffix="Test":  FooTest.java → Foo
    """
    name = PurePosixPath(file_path).stem
    if test_prefix and name.startswith(test_prefix):
        return name[len(test_prefix) :]
    if test_suffix and name.endswith(test_suffix):
        return name[: -len(test_suffix)]
    return None


def _find_source_file(
    base_name: str,
    test_path: str,
    repo_files: set[str],
    resolution: ImportResolution | None = None,
) -> str | None:
    """Find a source file matching *base_name*.

    Uses language-aware resolution when available, falling back
    to `.py` heuristics.
    """
    extensions = resolution.extensions if resolution else (".py",)
    source_roots = resolution.source_roots if resolution else ("", "src")

    # Direct matches: source_root/base_name + ext.
    for root in source_roots:
        for ext in extensions:
            candidate = f"{root}/{base_name}{ext}" if root else f"{base_name}{ext}"
            if candidate in repo_files:
                return candidate

    # Sibling of test directory: tests/test_foo → ../foo + ext.
    test_dir = PurePosixPath(test_path).parent
    if test_dir.name in ("tests", "test"):
        parent = test_dir.parent
        for root in ("", "src"):
            for ext in extensions:
                candidate = (
                    str(parent / root / f"{base_name}{ext}")
                    if root
                    else str(parent / f"{base_name}{ext}")
                )
                if candidate in repo_files:
                    return candidate

    # Underscore-to-slash expansion (Python convention: test_foo_bar → foo/bar).
    if "_" in base_name:
        nested = base_name.replace("_", "/")
        for root in source_roots:
            for ext in extensions:
                candidate = f"{root}/{nested}{ext}" if root else f"{nested}{ext}"
                if candidate in repo_files:
                    return candidate

    # Last resort: full-path suffix search, unique-match-only (same policy as
    # the import resolver's Tier 3 — drop rather than guess on ambiguity).
    candidates = _suffix_matches(base_name, repo_files, extensions)
    if len(candidates) == 1:
        return candidates[0]

    return None


def _imported_names_from_file(
    test_chunks: list[Chunk],
    source_file: str,
    repo_files: set[str],
    resolution: ImportResolution | None = None,
) -> list[str]:
    """Extract symbol names imported from *source_file* by the test.

    Reads `chunk.metadata` — no language-specific parsing.
    """
    names: list[str] = []
    for c in test_chunks:
        if c.kind != ChunkKind.IMPORT or not c.metadata:
            continue

        resolved = _resolve_import_to_file(c.metadata, c.file_path, repo_files, resolution)
        if resolved != source_file:
            continue

        names_str = c.metadata.names
        if names_str:
            names.extend(names_str.split(","))

    return names


def infer_test_edges(
    chunks: list[Chunk],
    repo_files: set[str],
    resolution_map: dict[str, ImportResolution] | None = None,
) -> list[Edge]:
    """Infer `TESTS` edges from test files.

    Links test functions to the source symbols they test, using
    file naming conventions and import analysis.  When
    *resolution_map* is provided, uses language-aware test-file
    detection and file resolution.
    """
    by_file: dict[str, list[Chunk]] = {}
    # Track per-file language for resolution lookup.
    file_language: dict[str, str] = {}
    for c in chunks:
        by_file.setdefault(c.file_path, []).append(c)
        if c.language and c.file_path not in file_language:
            file_language[c.file_path] = c.language

    symbol_index = _build_symbol_index(chunks)
    edges: list[Edge] = []

    for file_path, file_chunks in by_file.items():
        lang = file_language.get(file_path, "")
        resolution = resolution_map.get(lang) if resolution_map else None

        base_name = _strip_test_affix(
            file_path,
            test_prefix=resolution.test_prefix if resolution else "test_",
            test_suffix=resolution.test_suffix if resolution else "",
        )
        if base_name is None:
            continue

        source_file = _find_source_file(base_name, file_path, repo_files, resolution)
        if source_file is None:
            continue

        test_fns = [c for c in file_chunks if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD)]
        if not test_fns:
            continue

        imported_names = _imported_names_from_file(file_chunks, source_file, repo_files, resolution)

        if imported_names:
            for test_fn in test_fns:
                for name in imported_names:
                    target = symbol_index.get((source_file, name))
                    if target is not None:
                        edges.append(
                            Edge(source_id=test_fn.id, target_id=target.id, kind=EdgeKind.TESTS)
                        )
        else:
            source_symbols = [
                c
                for c in by_file.get(source_file, [])
                if c.kind in (ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD)
            ]
            if source_symbols:
                for test_fn in test_fns:
                    edges.append(
                        Edge(
                            source_id=test_fn.id,
                            target_id=source_symbols[0].id,
                            kind=EdgeKind.TESTS,
                        )
                    )

    return edges

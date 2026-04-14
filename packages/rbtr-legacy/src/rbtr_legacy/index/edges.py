"""Edge inference — imports, tests, docs relationships.

All public functions are pure: chunks in, edges out.  No I/O, no
store access.  The orchestration layer (Phase 5) feeds them chunks
and writes the results.

Two resolution strategies are used:

- **Structural (tree-sitter):** Import metadata is extracted by
  `treesitter.py` during parsing and stored in `chunk.metadata`.
  This gives exact module paths and symbol names.  Used for languages
  with a tree-sitter import extractor (currently Python).

- **Text search fallback:** When tree-sitter metadata is unavailable
  (unsupported language, doc sections, config files), we fall back to
  text matching — scanning chunk content for known symbol names or
  file-path fragments.  Less precise but works everywhere.

Doc↔code edges always use text search since docs are prose, not code.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from rbtr_legacy.index.models import Chunk, ChunkKind, Edge, EdgeKind, ImportMeta

# ── Shared helpers ───────────────────────────────────────────────────


def _resolve_module_to_file(
    module_path: str,
    repo_files: set[str],
) -> str | None:
    """Try to find a repo file matching *module_path*.

    Checks `module_path.py` and `module_path/__init__.py`.
    """
    candidates = [
        f"{module_path}.py",
        f"{module_path}/__init__.py",
    ]
    for candidate in candidates:
        if candidate in repo_files:
            return candidate
    return None


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


# ── Import edges ─────────────────────────────────────────────────────
#
# Strategy: structural (tree-sitter metadata) with text-search fallback.
#
# When tree-sitter metadata is present (`chunk.metadata` has `module`
# or `dots` keys), we use the exact module path and symbol names.
# When metadata is absent, we fall back to scanning the import chunk's
# text for known repo file stems — less precise but handles any language.


def _module_to_segments(module: str) -> list[str]:
    """Split a module string into path segments.

    Slash-separated paths (JS/TS/Go/Rust) are split on `/`.
    Dotted paths (Python/Java: `os.path`) are split on `.`.
    When the string has no separator, it is returned as a single
    segment.
    """
    if "/" in module:
        return [s for s in module.split("/") if s]
    if "." in module:
        return module.split(".")
    return [module]


def _resolve_relative_module(
    file_path: str,
    dots: int,
    module: str | None,
) -> str | None:
    """Resolve a relative import to a `/`-separated path prefix.

    *dots* is the number of levels up from the file: 1 = current
    directory, 2 = parent directory, etc.  This convention is shared
    across languages (Python `from .`, JS `./`, Rust `super::`).
    """
    parts = PurePosixPath(file_path).parts
    if len(parts) < dots:
        return None
    base_parts = parts[:-dots]
    if module:
        base_parts = (*base_parts, *_module_to_segments(module))
    return "/".join(base_parts) if base_parts else None


def _resolve_import_module(
    meta: ImportMeta,
    file_path: str,
) -> str | None:
    """Resolve import metadata to a `/`-separated module path."""
    dots_str = meta.get("dots")
    module = meta.get("module")

    if dots_str:
        return _resolve_relative_module(file_path, int(dots_str), module)
    if module:
        return "/".join(_module_to_segments(module))
    return None


def _structural_import_edges(
    imp: Chunk,
    symbol_index: dict[tuple[str, str], Chunk],
    all_chunks: list[Chunk],
    repo_files: set[str],
) -> list[Edge]:
    """Resolve import edges using tree-sitter metadata (exact)."""
    meta = imp.metadata
    module_path = _resolve_import_module(meta, imp.file_path)
    if module_path is None:
        return []

    target_file = _resolve_module_to_file(module_path, repo_files)
    if target_file is None:
        return []

    edges: list[Edge] = []
    names_str = meta.get("names")

    if names_str:
        # from foo import Bar, Baz → link to each named symbol.
        for name in names_str.split(","):
            target = symbol_index.get((target_file, name))
            if target is not None:
                edges.append(
                    Edge(
                        source_id=imp.id,
                        target_id=target.id,
                        kind=EdgeKind.IMPORTS,
                    )
                )
    else:
        # import foo → link to first symbol in the file.
        file_chunks = [
            c
            for c in all_chunks
            if c.file_path == target_file
            and c.kind
            in (
                ChunkKind.FUNCTION,
                ChunkKind.CLASS,
                ChunkKind.METHOD,
            )
        ]
        if file_chunks:
            edges.append(
                Edge(
                    source_id=imp.id,
                    target_id=file_chunks[0].id,
                    kind=EdgeKind.IMPORTS,
                )
            )

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
                edges.append(
                    Edge(
                        source_id=imp.id,
                        target_id=target.id,
                        kind=EdgeKind.IMPORTS,
                    )
                )

    return edges


def infer_import_edges(
    chunks: list[Chunk],
    repo_files: set[str],
) -> list[Edge]:
    """Infer `IMPORTS` edges from import chunks.

    Uses tree-sitter metadata when available (exact), falls back to
    text search otherwise (best-effort).
    """
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    symbol_index = _build_symbol_index(chunks)
    stem_index = _build_stem_index(repo_files)
    file_symbols = _build_file_symbol_index(symbol_index)
    edges: list[Edge] = []

    for imp in imports:
        has_metadata = bool(imp.metadata.get("module") or imp.metadata.get("dots"))

        if has_metadata:
            # Structural: tree-sitter gave us exact module/names.
            edges.extend(
                _structural_import_edges(
                    imp,
                    symbol_index,
                    chunks,
                    repo_files,
                )
            )
        else:
            # Text search fallback: scan for repo file stems.
            edges.extend(
                _text_search_import_edges(
                    imp,
                    file_symbols,
                    stem_index,
                )
            )

    return edges


# ── Test↔code edges ─────────────────────────────────────────────────
#
# Strategy: file-naming heuristic + import analysis (structural or
# text-search depending on what's available in the import chunks).


def _strip_test_prefix(file_path: str) -> str | None:
    """Extract the base name from a test file path.

    `tests/test_foo.py` → `foo`
    `test_bar.py` → `bar`
    `src/tests/test_baz_qux.py` → `baz_qux`
    """
    name = PurePosixPath(file_path).stem
    if not name.startswith("test_"):
        return None
    return name[5:]


def _find_source_file(
    base_name: str,
    test_path: str,
    repo_files: set[str],
) -> str | None:
    """Find a source file matching *base_name*."""
    candidates = [
        f"{base_name}.py",
        f"src/{base_name}.py",
    ]

    test_dir = PurePosixPath(test_path).parent
    if test_dir.name in ("tests", "test"):
        parent = test_dir.parent
        candidates.extend(
            [
                str(parent / f"{base_name}.py"),
                str(parent / "src" / f"{base_name}.py"),
            ]
        )

    if "_" in base_name:
        nested = base_name.replace("_", "/")
        candidates.extend(
            [
                f"{nested}.py",
                f"src/{nested}.py",
            ]
        )

    for candidate in candidates:
        if candidate in repo_files:
            return candidate

    suffix = f"/{base_name}.py"
    for f in sorted(repo_files):
        if f.endswith(suffix) or f == f"{base_name}.py":
            return f

    return None


def _imported_names_from_file(
    test_chunks: list[Chunk],
    source_file: str,
    repo_files: set[str],
) -> list[str]:
    """Extract symbol names imported from *source_file* by the test.

    Reads `chunk.metadata` — no language-specific parsing.
    """
    names: list[str] = []
    for c in test_chunks:
        if c.kind != ChunkKind.IMPORT or not c.metadata:
            continue

        module_path = _resolve_import_module(c.metadata, c.file_path)
        if module_path is None:
            continue

        resolved = _resolve_module_to_file(module_path, repo_files)
        if resolved != source_file:
            continue

        names_str = c.metadata.get("names")
        if names_str:
            names.extend(names_str.split(","))

    return names


def infer_test_edges(
    chunks: list[Chunk],
    repo_files: set[str],
) -> list[Edge]:
    """Infer `TESTS` edges from test files.

    Links test functions to the source symbols they test, using
    file naming conventions and import analysis.
    """
    by_file: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_file.setdefault(c.file_path, []).append(c)

    symbol_index = _build_symbol_index(chunks)
    edges: list[Edge] = []

    for file_path, file_chunks in by_file.items():
        base_name = _strip_test_prefix(file_path)
        if base_name is None:
            continue

        source_file = _find_source_file(base_name, file_path, repo_files)
        if source_file is None:
            continue

        test_fns = [c for c in file_chunks if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD)]
        if not test_fns:
            continue

        imported_names = _imported_names_from_file(file_chunks, source_file, repo_files)

        if imported_names:
            for test_fn in test_fns:
                for name in imported_names:
                    target = symbol_index.get((source_file, name))
                    if target is not None:
                        edges.append(
                            Edge(
                                source_id=test_fn.id,
                                target_id=target.id,
                                kind=EdgeKind.TESTS,
                            )
                        )
        else:
            source_symbols = [
                c
                for c in by_file.get(source_file, [])
                if c.kind
                in (
                    ChunkKind.FUNCTION,
                    ChunkKind.CLASS,
                    ChunkKind.METHOD,
                )
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


# ── Doc↔code edges ──────────────────────────────────────────────────
#
# Strategy: text search only.
#
# Docs are prose — no tree-sitter structure to leverage.  We scan
# DOC_SECTION content for word-boundary matches of known symbol names.
# This is inherently a text-search approach, not structural.

_MIN_NAME_LENGTH = 3


def infer_doc_edges(
    chunks: list[Chunk],
) -> list[Edge]:
    """Infer `DOCUMENTS` edges via text search.

    Tokenises `DOC_SECTION` content into words and checks set
    membership against known `FUNCTION`, `CLASS`, and `METHOD`
    names.  Skips names shorter than 3 characters to avoid false
    positives.

    This is a text-search-only approach — docs are prose, so
    tree-sitter structural analysis does not apply.
    """
    doc_chunks = [c for c in chunks if c.kind == ChunkKind.DOC_SECTION]
    code_chunks = [
        c for c in chunks if c.kind in (ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD)
    ]

    if not doc_chunks or not code_chunks:
        return []

    # Build name → chunks mapping (a name may appear in multiple files).
    name_to_chunks: dict[str, list[Chunk]] = {}
    for c in code_chunks:
        if len(c.name) >= _MIN_NAME_LENGTH:
            name_to_chunks.setdefault(c.name, []).append(c)

    known_names = set(name_to_chunks)

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()

    for doc in doc_chunks:
        # Tokenise once, intersect with known symbol names.
        doc_words = set(_WORD_RE.findall(doc.content))
        matched = doc_words & known_names
        for name in matched:
            for target in name_to_chunks[name]:
                pair = (doc.id, target.id)
                if pair not in seen:
                    seen.add(pair)
                    edges.append(
                        Edge(
                            source_id=doc.id,
                            target_id=target.id,
                            kind=EdgeKind.DOCUMENTS,
                        )
                    )

    return edges

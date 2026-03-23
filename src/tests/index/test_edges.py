"""Tests for edge inference — imports, tests, docs."""

from __future__ import annotations

from rbtr.index.edges import (
    _find_source_file,
    _resolve_module_to_file,
    _strip_test_prefix,
    infer_doc_edges,
    infer_import_edges,
    infer_test_edges,
)
from rbtr.index.models import Chunk, ChunkKind, EdgeKind, ImportMeta

# ── Helpers ──────────────────────────────────────────────────────────


def _chunk(
    *,
    chunk_id: str = "c1",
    file_path: str = "src/foo.py",
    kind: ChunkKind = ChunkKind.FUNCTION,
    name: str = "foo",
    content: str = "def foo(): ...",
    blob_sha: str = "sha1",
    line_start: int = 1,
    line_end: int = 1,
    scope: str = "",
    metadata: ImportMeta | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path=file_path,
        kind=kind,
        name=name,
        content=content,
        blob_sha=blob_sha,
        line_start=line_start,
        line_end=line_end,
        scope=scope,
        metadata=metadata or {},
    )


# ── _resolve_module_to_file ──────────────────────────────────────────


def test_resolve_direct_py() -> None:
    files = {"rbtr/index/models.py", "rbtr/index/__init__.py"}
    assert _resolve_module_to_file("rbtr/index/models", files) == "rbtr/index/models.py"


def test_resolve_init_py() -> None:
    files = {"rbtr/index/__init__.py"}
    assert _resolve_module_to_file("rbtr/index", files) == "rbtr/index/__init__.py"


def test_resolve_not_found() -> None:
    files = {"rbtr/other.py"}
    assert _resolve_module_to_file("rbtr/index/models", files) is None


# ── _strip_test_prefix ───────────────────────────────────────────────


def test_strip_test_prefix_simple() -> None:
    assert _strip_test_prefix("tests/test_foo.py") == "foo"


def test_strip_test_prefix_nested() -> None:
    assert _strip_test_prefix("src/tests/test_bar.py") == "bar"


def test_strip_test_prefix_no_prefix() -> None:
    assert _strip_test_prefix("src/foo.py") is None


def test_strip_test_prefix_root() -> None:
    assert _strip_test_prefix("test_baz.py") == "baz"


# ── _find_source_file ────────────────────────────────────────────────


def test_find_source_direct() -> None:
    files = {"foo.py", "bar.py"}
    assert _find_source_file("foo", "test_foo.py", files) == "foo.py"


def test_find_source_in_src() -> None:
    files = {"src/foo.py"}
    assert _find_source_file("foo", "tests/test_foo.py", files) == "src/foo.py"


def test_find_source_sibling_of_tests() -> None:
    files = {"lib/foo.py", "lib/tests/test_foo.py"}
    assert _find_source_file("foo", "lib/tests/test_foo.py", files) == "lib/foo.py"


def test_find_source_underscore_to_path() -> None:
    files = {"src/foo/bar.py"}
    assert _find_source_file("foo_bar", "tests/test_foo_bar.py", files) == "src/foo/bar.py"


def test_find_source_suffix_fallback() -> None:
    files = {"deep/nested/foo.py"}
    assert _find_source_file("foo", "tests/test_foo.py", files) == "deep/nested/foo.py"


def test_find_source_not_found() -> None:
    files = {"bar.py"}
    assert _find_source_file("foo", "tests/test_foo.py", files) is None


# ── infer_import_edges: structural (tree-sitter metadata) ────────────


def test_import_edges_structural_from_import() -> None:
    """Tree-sitter metadata: from src.models import User."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="from src.models import User",
        metadata={"module": "src.models", "names": "User"},
    )
    target = _chunk(
        chunk_id="user1",
        file_path="src/models.py",
        kind=ChunkKind.CLASS,
        name="User",
    )
    repo_files = {"src/app.py", "src/models.py"}
    edges = infer_import_edges([imp, target], repo_files)
    assert len(edges) == 1
    assert edges[0].source_id == "imp1"
    assert edges[0].target_id == "user1"
    assert edges[0].kind == EdgeKind.IMPORTS


def test_import_edges_structural_bare_import() -> None:
    """Tree-sitter metadata: import src.models."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="import src.models",
        metadata={"module": "src.models"},
    )
    target = _chunk(
        chunk_id="fn1",
        file_path="src/models.py",
        kind=ChunkKind.FUNCTION,
        name="create_user",
    )
    repo_files = {"src/app.py", "src/models.py"}
    edges = infer_import_edges([imp, target], repo_files)
    assert len(edges) == 1
    assert edges[0].target_id == "fn1"


def test_import_edges_structural_relative() -> None:
    """Tree-sitter metadata: from .models import Chunk."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/rbtr/index/store.py",
        kind=ChunkKind.IMPORT,
        name="from .models import Chunk",
        metadata={"dots": "1", "module": "models", "names": "Chunk"},
    )
    target = _chunk(
        chunk_id="chunk1",
        file_path="src/rbtr/index/models.py",
        kind=ChunkKind.CLASS,
        name="Chunk",
    )
    repo_files = {"src/rbtr/index/store.py", "src/rbtr/index/models.py"}
    edges = infer_import_edges([imp, target], repo_files)
    assert len(edges) == 1
    assert edges[0].target_id == "chunk1"


def test_import_edges_structural_relative_dot_only() -> None:
    """Tree-sitter metadata: from . import utils."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/rbtr/index/store.py",
        kind=ChunkKind.IMPORT,
        name="from . import utils",
        metadata={"dots": "1", "names": "utils"},
    )
    target = _chunk(
        chunk_id="utils1",
        file_path="src/rbtr/index/utils.py",
        kind=ChunkKind.FUNCTION,
        name="utils",
    )
    repo_files = {"src/rbtr/index/store.py", "src/rbtr/index/utils.py"}
    # dots=1 with no module → resolve to parent dir = src/rbtr/index
    # names=utils → look for "utils" symbol in src/rbtr/index.py or
    # src/rbtr/index/__init__.py — neither exists, so no edge.
    edges = infer_import_edges([imp, target], repo_files)
    assert edges == []


def test_import_edges_structural_stdlib_skipped() -> None:
    """Tree-sitter metadata: import os — no repo file matches."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="import os",
        metadata={"module": "os"},
    )
    repo_files = {"src/app.py"}
    edges = infer_import_edges([imp], repo_files)
    assert edges == []


def test_import_edges_structural_target_not_found() -> None:
    """Named import but symbol doesn't exist in target file."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="from src.models import Missing",
        metadata={"module": "src.models", "names": "Missing"},
    )
    target = _chunk(
        chunk_id="fn1",
        file_path="src/models.py",
        kind=ChunkKind.FUNCTION,
        name="existing",
    )
    repo_files = {"src/app.py", "src/models.py"}
    edges = infer_import_edges([imp, target], repo_files)
    assert edges == []


def test_import_edges_structural_multiple_names() -> None:
    """Tree-sitter metadata: from models import Chunk, Edge."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="from src.models import Chunk, Edge",
        metadata={"module": "src.models", "names": "Chunk,Edge"},
    )
    c1 = _chunk(chunk_id="c1", file_path="src/models.py", kind=ChunkKind.CLASS, name="Chunk")
    c2 = _chunk(chunk_id="c2", file_path="src/models.py", kind=ChunkKind.CLASS, name="Edge")
    repo_files = {"src/app.py", "src/models.py"}
    edges = infer_import_edges([imp, c1, c2], repo_files)
    assert len(edges) == 2
    target_ids = {e.target_id for e in edges}
    assert target_ids == {"c1", "c2"}


# ── infer_import_edges: text-search fallback ─────────────────────────


def test_import_edges_text_fallback_finds_file_stem() -> None:
    """No metadata → text search matches repo file stem in import text."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.rb",
        kind=ChunkKind.IMPORT,
        name="require 'models'",
        metadata={},  # No tree-sitter extractor for Ruby.
    )
    target = _chunk(
        chunk_id="fn1",
        file_path="src/models.rb",
        kind=ChunkKind.FUNCTION,
        name="create_user",
    )
    repo_files = {"src/app.rb", "src/models.rb"}
    edges = infer_import_edges([imp, target], repo_files)
    assert len(edges) == 1
    assert edges[0].source_id == "imp1"
    assert edges[0].target_id == "fn1"
    assert edges[0].kind == EdgeKind.IMPORTS


def test_import_edges_text_fallback_no_match() -> None:
    """No metadata + no file stem match → no edges."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.rb",
        kind=ChunkKind.IMPORT,
        name="require 'nonexistent'",
        metadata={},
    )
    repo_files = {"src/app.rb"}
    edges = infer_import_edges([imp], repo_files)
    assert edges == []


def test_import_edges_text_fallback_short_stem_skipped() -> None:
    """File stems < 3 chars are skipped in text search."""
    imp = _chunk(
        chunk_id="imp1",
        file_path="src/app.rb",
        kind=ChunkKind.IMPORT,
        name="require 'io'",
        metadata={},
    )
    target = _chunk(
        chunk_id="fn1",
        file_path="src/io.rb",
        kind=ChunkKind.FUNCTION,
        name="read",
    )
    repo_files = {"src/app.rb", "src/io.rb"}
    edges = infer_import_edges([imp, target], repo_files)
    assert edges == []


# ── infer_test_edges ─────────────────────────────────────────────────


def test_test_edges_with_import() -> None:
    src_fn = _chunk(
        chunk_id="fn1",
        file_path="src/foo.py",
        kind=ChunkKind.FUNCTION,
        name="do_stuff",
    )
    test_imp = _chunk(
        chunk_id="imp1",
        file_path="tests/test_foo.py",
        kind=ChunkKind.IMPORT,
        name="from src.foo import do_stuff",
        metadata={"module": "src.foo", "names": "do_stuff"},
    )
    test_fn = _chunk(
        chunk_id="tf1",
        file_path="tests/test_foo.py",
        kind=ChunkKind.FUNCTION,
        name="test_do_stuff",
    )
    repo_files = {"src/foo.py", "tests/test_foo.py"}
    edges = infer_test_edges([src_fn, test_imp, test_fn], repo_files)
    assert len(edges) == 1
    assert edges[0].source_id == "tf1"
    assert edges[0].target_id == "fn1"
    assert edges[0].kind == EdgeKind.TESTS


def test_test_edges_fallback_no_import() -> None:
    src_fn = _chunk(
        chunk_id="fn1",
        file_path="foo.py",
        kind=ChunkKind.FUNCTION,
        name="do_stuff",
    )
    test_fn = _chunk(
        chunk_id="tf1",
        file_path="test_foo.py",
        kind=ChunkKind.FUNCTION,
        name="test_do_stuff",
    )
    repo_files = {"foo.py", "test_foo.py"}
    edges = infer_test_edges([src_fn, test_fn], repo_files)
    assert len(edges) == 1
    assert edges[0].source_id == "tf1"
    assert edges[0].target_id == "fn1"


def test_test_edges_no_match() -> None:
    test_fn = _chunk(
        chunk_id="tf1",
        file_path="tests/test_foo.py",
        kind=ChunkKind.FUNCTION,
        name="test_stuff",
    )
    repo_files = {"tests/test_foo.py"}
    edges = infer_test_edges([test_fn], repo_files)
    assert edges == []


def test_test_edges_non_test_file_skipped() -> None:
    fn = _chunk(chunk_id="fn1", file_path="src/foo.py", name="foo")
    repo_files = {"src/foo.py"}
    edges = infer_test_edges([fn], repo_files)
    assert edges == []


def test_test_edges_multiple_test_fns() -> None:
    src_fn = _chunk(chunk_id="fn1", file_path="foo.py", name="do_stuff")
    tf1 = _chunk(
        chunk_id="tf1",
        file_path="test_foo.py",
        kind=ChunkKind.FUNCTION,
        name="test_a",
    )
    tf2 = _chunk(
        chunk_id="tf2",
        file_path="test_foo.py",
        kind=ChunkKind.FUNCTION,
        name="test_b",
    )
    repo_files = {"foo.py", "test_foo.py"}
    edges = infer_test_edges([src_fn, tf1, tf2], repo_files)
    assert len(edges) == 2
    target_ids = {e.target_id for e in edges}
    assert target_ids == {"fn1"}


# ── infer_doc_edges (text search only) ───────────────────────────────


def test_doc_edges_name_match() -> None:
    doc = _chunk(
        chunk_id="doc1",
        file_path="README.md",
        kind=ChunkKind.DOC_SECTION,
        name="Usage",
        content="Call `do_stuff` to process the data.",
    )
    fn = _chunk(
        chunk_id="fn1",
        file_path="src/foo.py",
        kind=ChunkKind.FUNCTION,
        name="do_stuff",
        content="def do_stuff(): ...",
    )
    edges = infer_doc_edges([doc, fn])
    assert len(edges) == 1
    assert edges[0].source_id == "doc1"
    assert edges[0].target_id == "fn1"
    assert edges[0].kind == EdgeKind.DOCUMENTS


def test_doc_edges_short_name_skipped() -> None:
    doc = _chunk(
        chunk_id="doc1",
        file_path="README.md",
        kind=ChunkKind.DOC_SECTION,
        name="Info",
        content="Use fn to process.",
    )
    fn = _chunk(
        chunk_id="fn1",
        file_path="src/foo.py",
        kind=ChunkKind.FUNCTION,
        name="fn",
        content="def fn(): ...",
    )
    edges = infer_doc_edges([doc, fn])
    assert edges == []


def test_doc_edges_word_boundary() -> None:
    doc = _chunk(
        chunk_id="doc1",
        file_path="README.md",
        kind=ChunkKind.DOC_SECTION,
        name="Ref",
        content="The processing step runs daily.",
    )
    fn = _chunk(
        chunk_id="fn1",
        file_path="src/foo.py",
        kind=ChunkKind.FUNCTION,
        name="process",
        content="def process(): ...",
    )
    edges = infer_doc_edges([doc, fn])
    # "processing" should NOT match "process" at word boundary.
    assert edges == []


def test_doc_edges_no_docs() -> None:
    fn = _chunk(chunk_id="fn1", name="foo")
    edges = infer_doc_edges([fn])
    assert edges == []


def test_doc_edges_no_code() -> None:
    doc = _chunk(
        chunk_id="doc1",
        kind=ChunkKind.DOC_SECTION,
        name="Readme",
        content="Hello world.",
    )
    edges = infer_doc_edges([doc])
    assert edges == []


def test_doc_edges_dedup() -> None:
    doc = _chunk(
        chunk_id="doc1",
        file_path="README.md",
        kind=ChunkKind.DOC_SECTION,
        name="Usage",
        content="Call do_stuff then do_stuff again.",
    )
    fn = _chunk(
        chunk_id="fn1",
        file_path="src/foo.py",
        kind=ChunkKind.FUNCTION,
        name="do_stuff",
    )
    edges = infer_doc_edges([doc, fn])
    assert len(edges) == 1


def test_doc_edges_multiple_symbols() -> None:
    doc = _chunk(
        chunk_id="doc1",
        file_path="README.md",
        kind=ChunkKind.DOC_SECTION,
        name="API",
        content="Use create_user and delete_user.",
    )
    fn1 = _chunk(
        chunk_id="fn1",
        file_path="src/api.py",
        kind=ChunkKind.FUNCTION,
        name="create_user",
    )
    fn2 = _chunk(
        chunk_id="fn2",
        file_path="src/api.py",
        kind=ChunkKind.FUNCTION,
        name="delete_user",
    )
    edges = infer_doc_edges([doc, fn1, fn2])
    assert len(edges) == 2
    target_ids = {e.target_id for e in edges}
    assert target_ids == {"fn1", "fn2"}

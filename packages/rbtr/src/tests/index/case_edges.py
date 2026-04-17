"""Scenarios for the edge-inference functions.

Each case returns an ``EdgeCase`` describing the chunks present,
the repo's file set, the inference function to call, and the
expected edge IDs.  Tests in ``test_edges_inference.py`` dispatch
on ``EdgeCase.fn`` to the matching ``infer_*`` function and assert
the result.

Chunks are declared as ``ChunkSpec`` tuples of the fields that
matter for edge inference.  The fixture fills in pydantic-required
boilerplate (``blob_sha``, ``line_start``, ``line_end``, empty
``content`` when unused).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rbtr.index.models import ChunkKind

from tests.index.cases_common import ChunkSpec


class InferFn(StrEnum):
    IMPORT = "import"
    TEST = "test"
    DOC = "doc"

@dataclass(frozen=True)
class EdgeCase:
    fn: InferFn
    chunks: list[ChunkSpec]
    repo_files: frozenset[str] = frozenset()
    # Expected set of (source_id, target_id) pairs in the result,
    # regardless of order.  Empty = no edges expected.
    expected: frozenset[tuple[str, str]] = frozenset()


# ── import edges: structural (tree-sitter metadata) ─────────────────


def case_import_from_import() -> EdgeCase:
    """from src.models import User — module + names → exact target."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import User",
                file_path="src/app.py",
                metadata={"module": "src.models", "names": "User"},
            ),
            ChunkSpec(
                id="user1",
                kind=ChunkKind.CLASS,
                name="User",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        expected=frozenset({("imp1", "user1")}),
    )


def case_import_bare_module() -> EdgeCase:
    """import src.models — module only → any symbol in target file."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import src.models",
                file_path="src/app.py",
                metadata={"module": "src.models"},
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="create_user",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        expected=frozenset({("imp1", "fn1")}),
    )


def case_import_relative() -> EdgeCase:
    """from .models import Chunk — dots=1 + module + names."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from .models import Chunk",
                file_path="src/rbtr/index/store.py",
                metadata={"dots": "1", "module": "models", "names": "Chunk"},
            ),
            ChunkSpec(
                id="chunk1",
                kind=ChunkKind.CLASS,
                name="Chunk",
                file_path="src/rbtr/index/models.py",
            ),
        ],
        repo_files=frozenset(
            {"src/rbtr/index/store.py", "src/rbtr/index/models.py"}
        ),
        expected=frozenset({("imp1", "chunk1")}),
    )


def case_import_relative_dot_only_unresolved() -> EdgeCase:
    """from . import utils — dots=1 only; package __init__ missing."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from . import utils",
                file_path="src/rbtr/index/store.py",
                metadata={"dots": "1", "names": "utils"},
            ),
            ChunkSpec(
                id="utils1",
                kind=ChunkKind.FUNCTION,
                name="utils",
                file_path="src/rbtr/index/utils.py",
            ),
        ],
        repo_files=frozenset(
            {"src/rbtr/index/store.py", "src/rbtr/index/utils.py"}
        ),
        expected=frozenset(),  # no package file resolves
    )


def case_import_stdlib_module_skipped() -> EdgeCase:
    """import os — module doesn't resolve to a repo file."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import os",
                file_path="src/app.py",
                metadata={"module": "os"},
            ),
        ],
        repo_files=frozenset({"src/app.py"}),
        expected=frozenset(),
    )


def case_import_target_symbol_missing() -> EdgeCase:
    """Named import but symbol doesn't exist in target file."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import Missing",
                file_path="src/app.py",
                metadata={"module": "src.models", "names": "Missing"},
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="existing",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        expected=frozenset(),
    )


def case_import_multiple_names() -> EdgeCase:
    """from src.models import Chunk, Edge — names list parsed."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import Chunk, Edge",
                file_path="src/app.py",
                metadata={"module": "src.models", "names": "Chunk,Edge"},
            ),
            ChunkSpec(
                id="chunk1",
                kind=ChunkKind.CLASS,
                name="Chunk",
                file_path="src/models.py",
            ),
            ChunkSpec(
                id="edge1",
                kind=ChunkKind.CLASS,
                name="Edge",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        expected=frozenset({("imp1", "chunk1"), ("imp1", "edge1")}),
    )


# ── import edges: text-search fallback ──────────────────────────────


def case_import_text_fallback_file_stem_match() -> EdgeCase:
    """No metadata → text search matches repo file stem."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="require 'models'",
                file_path="src/app.rb",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="create_user",
                file_path="src/models.rb",
            ),
        ],
        repo_files=frozenset({"src/app.rb", "src/models.rb"}),
        expected=frozenset({("imp1", "fn1")}),
    )


def case_import_text_fallback_no_match() -> EdgeCase:
    """No metadata + no file stem match."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="require 'nonexistent'",
                file_path="src/app.rb",
            ),
        ],
        repo_files=frozenset({"src/app.rb"}),
        expected=frozenset(),
    )


def case_import_text_fallback_short_stem_skipped() -> EdgeCase:
    """File stems shorter than 3 chars are ignored by text search."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="require 'io'",
                file_path="src/app.rb",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="read",
                file_path="src/io.rb",
            ),
        ],
        repo_files=frozenset({"src/app.rb", "src/io.rb"}),
        expected=frozenset(),
    )


# ── import edges: coverage gaps ─────────────────────────────────────


def case_import_relative_overflow() -> EdgeCase:
    """Relative import with more dots than path depth."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from .....deep import thing",
                file_path="a/b.py",
                metadata={"dots": "5", "module": "deep", "names": "thing"},
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="thing",
                file_path="deep.py",
            ),
        ],
        repo_files=frozenset({"a/b.py", "deep.py"}),
        expected=frozenset(),
    )


def case_import_metadata_without_module_or_dots() -> EdgeCase:
    """Metadata only has ``names`` — no way to resolve."""
    return EdgeCase(
        fn=InferFn.IMPORT,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import something",
                file_path="src/app.py",
                metadata={"names": "Foo"},
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="Foo",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/foo.py"}),
        expected=frozenset(),
    )


# ── test edges ──────────────────────────────────────────────────────


def case_test_edges_with_import() -> EdgeCase:
    """Test file imports source — edge links test_fn to imported symbol."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.foo import do_stuff",
                file_path="tests/test_foo.py",
                metadata={"module": "src.foo", "names": "do_stuff"},
            ),
            ChunkSpec(
                id="tf1",
                kind=ChunkKind.FUNCTION,
                name="test_do_stuff",
                file_path="tests/test_foo.py",
            ),
        ],
        repo_files=frozenset({"src/foo.py", "tests/test_foo.py"}),
        expected=frozenset({("tf1", "fn1")}),
    )


def case_test_edges_fallback_no_import() -> EdgeCase:
    """No import in test file — fall back to file-name convention."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="foo.py",
            ),
            ChunkSpec(
                id="tf1",
                kind=ChunkKind.FUNCTION,
                name="test_do_stuff",
                file_path="test_foo.py",
            ),
        ],
        repo_files=frozenset({"foo.py", "test_foo.py"}),
        expected=frozenset({("tf1", "fn1")}),
    )


def case_test_edges_no_source_file() -> EdgeCase:
    """Test file exists but the source it would test doesn't."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="tf1",
                kind=ChunkKind.FUNCTION,
                name="test_stuff",
                file_path="tests/test_foo.py",
            ),
        ],
        repo_files=frozenset({"tests/test_foo.py"}),
        expected=frozenset(),
    )


def case_test_edges_non_test_file_skipped() -> EdgeCase:
    """A plain source file never yields test edges."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="foo",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"src/foo.py"}),
        expected=frozenset(),
    )


def case_test_edges_multiple_test_fns() -> EdgeCase:
    """Several tests in the same file all link to the same source fn."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="foo.py",
            ),
            ChunkSpec(
                id="tf1",
                kind=ChunkKind.FUNCTION,
                name="test_a",
                file_path="test_foo.py",
            ),
            ChunkSpec(
                id="tf2",
                kind=ChunkKind.FUNCTION,
                name="test_b",
                file_path="test_foo.py",
            ),
        ],
        repo_files=frozenset({"foo.py", "test_foo.py"}),
        expected=frozenset({("tf1", "fn1"), ("tf2", "fn1")}),
    )


def case_test_edges_imports_without_functions() -> EdgeCase:
    """Test file has only imports, no test functions — no edges."""
    return EdgeCase(
        fn=InferFn.TEST,
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.foo import do_stuff",
                file_path="tests/test_foo.py",
                metadata={"module": "src.foo", "names": "do_stuff"},
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"src/foo.py", "tests/test_foo.py"}),
        expected=frozenset(),
    )


# ── doc edges ───────────────────────────────────────────────────────


def case_doc_edges_name_match() -> EdgeCase:
    """Doc mentions `do_stuff` → edge to the function with that name."""
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Usage",
                file_path="README.md",
                content="Call `do_stuff` to process the data.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
                content="def do_stuff(): ...",
            ),
        ],
        expected=frozenset({("doc1", "fn1")}),
    )


def case_doc_edges_short_name_skipped() -> EdgeCase:
    """Doc mentions a 2-char symbol — ignored (too short)."""
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Info",
                file_path="README.md",
                content="Use fn to process.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="fn",
                file_path="src/foo.py",
                content="def fn(): ...",
            ),
        ],
        expected=frozenset(),
    )


def case_doc_edges_word_boundary() -> EdgeCase:
    """'processing' must NOT match 'process' via substring."""
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Ref",
                file_path="README.md",
                content="The processing step runs daily.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="process",
                file_path="src/foo.py",
                content="def process(): ...",
            ),
        ],
        expected=frozenset(),
    )


def case_doc_edges_no_docs_yields_nothing() -> EdgeCase:
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(id="fn1", kind=ChunkKind.FUNCTION, name="foo"),
        ],
        expected=frozenset(),
    )


def case_doc_edges_no_code_yields_nothing() -> EdgeCase:
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Readme",
                content="Hello world.",
            ),
        ],
        expected=frozenset(),
    )


def case_doc_edges_dedup_repeated_mention() -> EdgeCase:
    """Doc mentions the same symbol twice — only one edge."""
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Usage",
                file_path="README.md",
                content="Call do_stuff then do_stuff again.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
        ],
        expected=frozenset({("doc1", "fn1")}),
    )


def case_doc_edges_multiple_symbols() -> EdgeCase:
    """Doc mentions two different symbols — one edge each."""
    return EdgeCase(
        fn=InferFn.DOC,
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="API",
                file_path="README.md",
                content="Use create_user and delete_user.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="create_user",
                file_path="src/api.py",
            ),
            ChunkSpec(
                id="fn2",
                kind=ChunkKind.FUNCTION,
                name="delete_user",
                file_path="src/api.py",
            ),
        ],
        expected=frozenset({("doc1", "fn1"), ("doc1", "fn2")}),
    )

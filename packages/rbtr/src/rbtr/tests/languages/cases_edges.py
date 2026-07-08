"""Scenarios for the edge-inference functions.

Each case returns an `EdgeScenario` describing the chunks present,
the repo's file set, and the expected edge IDs.  Tests in
`test_edge_inference.py` run `infer_import_edges` and assert the
result.

Chunks are declared as `ChunkSpec` tuples of the fields that
matter for edge inference.  The fixture fills in pydantic-required
boilerplate (`blob_sha`, `line_start`, `line_end`, empty
`content` when unused).
"""

from __future__ import annotations

from dataclasses import dataclass

from rbtr.index.models import ChunkKind, EdgeKind, ImportMeta
from rbtr.languages.edges import ImportResolution
from rbtr.languages.registration import ModuleStyle

from ..index.cases_common import ChunkSpec


@dataclass(frozen=True)
class EdgeScenario:
    chunks: list[ChunkSpec]
    repo_files: frozenset[str] = frozenset()
    # Expected set of (source_id, target_id) pairs in the result,
    # regardless of order.  Empty = no edges expected.
    expected: frozenset[tuple[str, str]] = frozenset()
    resolution_map: dict[str, ImportResolution] | None = None
    # Override the default edge-kind check (IMPORT→IMPORTS, etc.).
    # Used when prose-language imports produce DOCUMENTS edges.
    expected_edge_kind: EdgeKind | None = None


_PYTHON_RESOLUTION = ImportResolution(
    extensions=(".py", ".pyi"),
    index_files=("__init__.py",),
    source_roots=("", "src"),
    path_substitutions=(),
    module_style=ModuleStyle.DOTTED,
)
_PYTHON_MAP: dict[str, ImportResolution] = {"python": _PYTHON_RESOLUTION}


# ── import edges: structural (tree-sitter metadata) ─────────────────


def case_import_from_import() -> EdgeScenario:
    """from src.models import User — module + names → exact target."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import User",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.models", names="User"),
            ),
            ChunkSpec(
                id="user1",
                kind=ChunkKind.CLASS,
                name="User",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "user1")}),
    )


def case_import_variable_target() -> EdgeScenario:
    """from src.config import config — named import of a module-level VARIABLE."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp_cfg",
                kind=ChunkKind.IMPORT,
                name="from src.config import config",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.config", names="config"),
            ),
            ChunkSpec(
                id="var_cfg",
                kind=ChunkKind.VARIABLE,
                name="config",
                file_path="src/config.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/config.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp_cfg", "var_cfg")}),
    )


def case_import_destructured_variable_target() -> EdgeScenario:
    """from src.config import a — `a` came from `a, b = load()`, still links."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp_a",
                kind=ChunkKind.IMPORT,
                name="from src.config import a",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.config", names="a"),
            ),
            ChunkSpec(
                id="var_a",
                kind=ChunkKind.VARIABLE,
                name="a",
                file_path="src/config.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/config.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp_a", "var_a")}),
    )


def case_import_bare_module() -> EdgeScenario:
    """import src.models — bare import → edges to all non-import chunks."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import src.models",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.models"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="create_user",
                file_path="src/models.py",
            ),
            ChunkSpec(
                id="cls1",
                kind=ChunkKind.CLASS,
                name="User",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "fn1"), ("imp1", "cls1")}),
    )


def case_import_relative() -> EdgeScenario:
    """from .models import Chunk — dots=1 + module + names."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from .models import Chunk",
                file_path="src/rbtr/index/store.py",
                language="python",
                metadata=ImportMeta(dots="1", module="models", names="Chunk"),
            ),
            ChunkSpec(
                id="chunk1",
                kind=ChunkKind.CLASS,
                name="Chunk",
                file_path="src/rbtr/index/models.py",
            ),
        ],
        repo_files=frozenset({"src/rbtr/index/store.py", "src/rbtr/index/models.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "chunk1")}),
    )


def case_import_relative_dot_only_unresolved() -> EdgeScenario:
    """from . import utils — dots=1 only; package __init__ missing."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from . import utils",
                file_path="src/rbtr/index/store.py",
                language="python",
                metadata=ImportMeta(dots="1", names="utils"),
            ),
            ChunkSpec(
                id="utils1",
                kind=ChunkKind.FUNCTION,
                name="utils",
                file_path="src/rbtr/index/utils.py",
            ),
        ],
        repo_files=frozenset({"src/rbtr/index/store.py", "src/rbtr/index/utils.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),  # no package file resolves
    )


def case_import_stdlib_module_skipped() -> EdgeScenario:
    """import os — module doesn't resolve to a repo file."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import os",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="os"),
            ),
        ],
        repo_files=frozenset({"src/app.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


def case_import_target_symbol_missing() -> EdgeScenario:
    """Named import but symbol doesn't exist in target file."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import Missing",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.models", names="Missing"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="existing",
                file_path="src/models.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/models.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


def case_import_multiple_names() -> EdgeScenario:
    """from src.models import Chunk, Edge — names list parsed."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from src.models import Chunk, Edge",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(module="src.models", names="Chunk,Edge"),
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
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "chunk1"), ("imp1", "edge1")}),
    )


def case_import_monorepo_absolute_suffix() -> EdgeScenario:
    """Absolute import resolves across a packages/*/src layout via suffix."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from rbtr.index.store import IndexStore",
                file_path="packages/app/src/app/main.py",
                language="python",
                metadata=ImportMeta(module="rbtr.index.store", names="IndexStore"),
            ),
            ChunkSpec(
                id="cls1",
                kind=ChunkKind.CLASS,
                name="IndexStore",
                file_path="packages/rbtr/src/rbtr/index/store.py",
            ),
        ],
        repo_files=frozenset(
            {
                "packages/app/src/app/main.py",
                "packages/rbtr/src/rbtr/index/store.py",
            }
        ),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "cls1")}),
    )


def case_import_suffix_helpers_non_collision() -> EdgeScenario:
    """Full-path suffix resolves to the right file, not a same-named stem."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from rbtr.utils.helpers import format_path",
                file_path="packages/app/src/app/main.py",
                language="python",
                metadata=ImportMeta(module="rbtr.utils.helpers", names="format_path"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="format_path",
                file_path="packages/rbtr/src/rbtr/utils/helpers.py",
            ),
            ChunkSpec(
                id="fn2",
                kind=ChunkKind.FUNCTION,
                name="format_path",
                file_path="other/helpers.py",
            ),
        ],
        repo_files=frozenset(
            {
                "packages/app/src/app/main.py",
                "packages/rbtr/src/rbtr/utils/helpers.py",
                "other/helpers.py",
            }
        ),
        resolution_map=_PYTHON_MAP,
        expected=frozenset({("imp1", "fn1")}),
    )


def case_import_suffix_collision_dropped() -> EdgeScenario:
    """Same full-path suffix in two packages is ambiguous → no edge."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from common.io import read_blob",
                file_path="packages/a/src/a/main.py",
                language="python",
                metadata=ImportMeta(module="common.io", names="read_blob"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="read_blob",
                file_path="packages/a/src/common/io.py",
            ),
            ChunkSpec(
                id="fn2",
                kind=ChunkKind.FUNCTION,
                name="read_blob",
                file_path="packages/b/src/common/io.py",
            ),
        ],
        repo_files=frozenset(
            {
                "packages/a/src/a/main.py",
                "packages/a/src/common/io.py",
                "packages/b/src/common/io.py",
            }
        ),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


def case_import_suffix_single_segment_guard() -> EdgeScenario:
    """A single-segment module never matches a nested file via suffix."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import models",
                file_path="packages/a/src/a/main.py",
                language="python",
                metadata=ImportMeta(module="models"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="build",
                file_path="deep/nested/models.py",
            ),
        ],
        repo_files=frozenset({"packages/a/src/a/main.py", "deep/nested/models.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


# ── import edges: text-search fallback ──────────────────────────────


def case_import_text_fallback_file_stem_match() -> EdgeScenario:
    """No metadata → text search matches repo file stem."""
    return EdgeScenario(
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


def case_import_text_fallback_no_match() -> EdgeScenario:
    """No metadata + no file stem match."""
    return EdgeScenario(
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


def case_import_text_fallback_short_stem_skipped() -> EdgeScenario:
    """File stems shorter than 3 chars are ignored by text search."""
    return EdgeScenario(
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


def case_import_relative_overflow() -> EdgeScenario:
    """Relative import with more dots than path depth."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="from .....deep import thing",
                file_path="a/b.py",
                language="python",
                metadata=ImportMeta(dots="5", module="deep", names="thing"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="thing",
                file_path="deep.py",
            ),
        ],
        repo_files=frozenset({"a/b.py", "deep.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


def case_import_metadata_without_module_or_dots() -> EdgeScenario:
    """Metadata only has `names` — no way to resolve."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import something",
                file_path="src/app.py",
                language="python",
                metadata=ImportMeta(names="Foo"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="Foo",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"src/app.py", "src/foo.py"}),
        resolution_map=_PYTHON_MAP,
        expected=frozenset(),
    )


# ── doc edges ───────────────────────────────────────────────────────


_DOC_RESOLUTION: dict[str, ImportResolution] = {
    "markdown": ImportResolution(
        extensions=(".md", ".rst", ".py"),
        index_files=(),
        source_roots=("",),
        path_substitutions=(),
    ),
    "rst": ImportResolution(
        extensions=(".rst", ".md", ".py"),
        index_files=(),
        source_roots=("",),
        path_substitutions=(),
    ),
}


def case_doc_md_link_to_code() -> EdgeScenario:
    """MD [link](../src/foo.py) → DOCUMENTS edges to all symbols."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="[source](../src/foo.py)",
                file_path="docs/guide.md",
                language="markdown",
                metadata=ImportMeta(module="../src/foo.py"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
            ChunkSpec(
                id="fn2",
                kind=ChunkKind.FUNCTION,
                name="other",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"docs/guide.md", "src/foo.py"}),
        resolution_map=_DOC_RESOLUTION,
        expected=frozenset({("imp1", "fn1"), ("imp1", "fn2")}),
        expected_edge_kind=EdgeKind.DOCUMENTS,
    )


def case_doc_md_link_to_doc() -> EdgeScenario:
    """MD [link](other.md) → DOCUMENTS edges to all DOC_SECTIONs."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="[guide](other.md)",
                file_path="docs/index.md",
                language="markdown",
                metadata=ImportMeta(module="./other.md"),
            ),
            ChunkSpec(
                id="sec1",
                kind=ChunkKind.DOC_SECTION,
                name="Intro",
                file_path="docs/other.md",
                language="markdown",
            ),
            ChunkSpec(
                id="sec2",
                kind=ChunkKind.DOC_SECTION,
                name="Details",
                file_path="docs/other.md",
                language="markdown",
            ),
        ],
        repo_files=frozenset({"docs/index.md", "docs/other.md"}),
        resolution_map=_DOC_RESOLUTION,
        expected=frozenset({("imp1", "sec1"), ("imp1", "sec2")}),
        expected_edge_kind=EdgeKind.DOCUMENTS,
    )


def case_doc_md_link_with_fragment() -> EdgeScenario:
    """MD [link](other.md#details) → targeted DOCUMENTS edge."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="[ref](other.md#details)",
                file_path="docs/index.md",
                language="markdown",
                metadata=ImportMeta(module="./other.md", names="details"),
            ),
            ChunkSpec(
                id="sec1",
                kind=ChunkKind.DOC_SECTION,
                name="Intro",
                file_path="docs/other.md",
                language="markdown",
            ),
            ChunkSpec(
                id="sec2",
                kind=ChunkKind.DOC_SECTION,
                name="details",
                file_path="docs/other.md",
                language="markdown",
            ),
        ],
        repo_files=frozenset({"docs/index.md", "docs/other.md"}),
        resolution_map=_DOC_RESOLUTION,
        expected=frozenset({("imp1", "sec2")}),
        expected_edge_kind=EdgeKind.DOCUMENTS,
    )


def case_doc_rst_func_role() -> EdgeScenario:
    """RST :func:`do_stuff` → targeted DOCUMENTS edge to function."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name=":func:`do_stuff`",
                file_path="docs/api.rst",
                language="rst",
                metadata=ImportMeta(names="do_stuff"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
        ],
        repo_files=frozenset({"docs/api.rst", "src/foo.py"}),
        resolution_map=_DOC_RESOLUTION,
        expected=frozenset({("imp1", "fn1")}),
        expected_edge_kind=EdgeKind.DOCUMENTS,
    )


def case_doc_rst_doc_role() -> EdgeScenario:
    """RST :doc:`api/module` → DOCUMENTS edges to all chunks."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name=":doc:`api/module`",
                file_path="docs/index.rst",
                language="rst",
                metadata=ImportMeta(module="api/module"),
            ),
            ChunkSpec(
                id="sec1",
                kind=ChunkKind.DOC_SECTION,
                name="API",
                file_path="api/module.rst",
                language="rst",
            ),
        ],
        repo_files=frozenset({"docs/index.rst", "api/module.rst"}),
        resolution_map=_DOC_RESOLUTION,
        expected=frozenset({("imp1", "sec1")}),
        expected_edge_kind=EdgeKind.DOCUMENTS,
    )


def case_doc_md_bare_mention_no_edge() -> EdgeScenario:
    """Prose mentions do_stuff without a link → no edge."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Usage",
                file_path="README.md",
                language="markdown",
                content="Call do_stuff to process the data.",
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="do_stuff",
                file_path="src/foo.py",
            ),
        ],
        expected=frozenset(),
    )


def case_doc_no_references_no_edges() -> EdgeScenario:
    """DOC_SECTION only, no IMPORT chunks → no edges."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="doc1",
                kind=ChunkKind.DOC_SECTION,
                name="Readme",
                file_path="README.md",
                language="markdown",
                content="Hello world.",
            ),
        ],
        expected=frozenset(),
    )


# ── cross-language import edges ─────────────────────────────────────


def case_import_html_script_src_relative() -> EdgeScenario:
    """HTML <script src="./app.js"> → JS function via language_hint."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name='<script src="./app.js"></script>',
                file_path="site/index.html",
                language="html",
                metadata=ImportMeta(module="./app.js", language_hint="javascript"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="init",
                file_path="site/app.js",
                language="javascript",
            ),
        ],
        repo_files=frozenset({"site/index.html", "site/app.js"}),
        resolution_map={
            "javascript": ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        expected=frozenset({("imp1", "fn1")}),
    )


def case_import_html_script_src_parent() -> EdgeScenario:
    """HTML <script src="../lib/core.js"> → all chunks in JS file via ../."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name='<script src="../lib/core.js"></script>',
                file_path="site/pages/index.html",
                language="html",
                metadata=ImportMeta(module="../lib/core.js", language_hint="javascript"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="core_init",
                file_path="site/lib/core.js",
                language="javascript",
            ),
        ],
        repo_files=frozenset({"site/pages/index.html", "site/lib/core.js"}),
        resolution_map={
            "javascript": ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        expected=frozenset({("imp1", "fn1")}),
    )


def case_import_js_css_via_language_hint() -> EdgeScenario:
    """JS import './styles.css' → edges to all DOC_SECTION chunks in CSS file."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="import './styles.css'",
                file_path="src/app.js",
                language="javascript",
                metadata=ImportMeta(module="styles", dots="1"),
            ),
            ChunkSpec(
                id="css1",
                kind=ChunkKind.DOC_SECTION,
                name="body",
                file_path="src/styles.css",
                language="css",
            ),
            ChunkSpec(
                id="css2",
                kind=ChunkKind.DOC_SECTION,
                name=".header",
                file_path="src/styles.css",
                language="css",
            ),
            ChunkSpec(
                id="css3",
                kind=ChunkKind.DOC_SECTION,
                name=".footer",
                file_path="src/styles.css",
                language="css",
            ),
        ],
        repo_files=frozenset({"src/app.js", "src/styles.css"}),
        resolution_map={
            "javascript": ImportResolution(
                extensions=(".js", ".css"),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        expected=frozenset({("imp1", "css1"), ("imp1", "css2"), ("imp1", "css3")}),
    )


def case_import_html_link_href_css() -> EdgeScenario:
    """HTML <link href="styles.css"> → edges to all chunks in CSS file."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name='<link href="styles.css">',
                file_path="site/index.html",
                language="html",
                metadata=ImportMeta(module="./styles.css", language_hint="css"),
            ),
            ChunkSpec(
                id="css1",
                kind=ChunkKind.DOC_SECTION,
                name="body",
                file_path="site/styles.css",
                language="css",
            ),
            ChunkSpec(
                id="css2",
                kind=ChunkKind.DOC_SECTION,
                name=".nav",
                file_path="site/styles.css",
                language="css",
            ),
        ],
        repo_files=frozenset({"site/index.html", "site/styles.css"}),
        resolution_map={
            "css": ImportResolution(
                extensions=(".css",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        expected=frozenset({("imp1", "css1"), ("imp1", "css2")}),
    )


def case_import_bare_excludes_import_chunks() -> EdgeScenario:
    """Bare import fans out to all chunks except IMPORT chunks in target."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name='@import "base.css"',
                file_path="site/main.css",
                language="css",
                metadata=ImportMeta(module="./styles.css"),
            ),
            ChunkSpec(
                id="css_imp",
                kind=ChunkKind.IMPORT,
                name='@import "reset.css"',
                file_path="site/styles.css",
                language="css",
                metadata=ImportMeta(module="./reset.css"),
            ),
            ChunkSpec(
                id="css1",
                kind=ChunkKind.DOC_SECTION,
                name="body",
                file_path="site/styles.css",
                language="css",
            ),
        ],
        repo_files=frozenset({"site/main.css", "site/styles.css"}),
        resolution_map={
            "css": ImportResolution(
                extensions=(".css",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        # css_imp is an IMPORT chunk → excluded from fan-out.
        expected=frozenset({("imp1", "css1")}),
    )


def case_import_bash_source() -> EdgeScenario:
    """source ./lib/utils.sh → edges to all functions in target."""
    return EdgeScenario(
        chunks=[
            ChunkSpec(
                id="imp1",
                kind=ChunkKind.IMPORT,
                name="source ./lib/utils.sh",
                file_path="scripts/deploy.sh",
                language="bash",
                metadata=ImportMeta(module="./lib/utils.sh"),
            ),
            ChunkSpec(
                id="fn1",
                kind=ChunkKind.FUNCTION,
                name="log_info",
                file_path="scripts/lib/utils.sh",
                language="bash",
            ),
            ChunkSpec(
                id="fn2",
                kind=ChunkKind.FUNCTION,
                name="log_error",
                file_path="scripts/lib/utils.sh",
                language="bash",
            ),
        ],
        repo_files=frozenset({"scripts/deploy.sh", "scripts/lib/utils.sh"}),
        resolution_map={
            "bash": ImportResolution(
                extensions=(".sh", ".bash", ".zsh"),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
        },
        expected=frozenset({("imp1", "fn1"), ("imp1", "fn2")}),
    )

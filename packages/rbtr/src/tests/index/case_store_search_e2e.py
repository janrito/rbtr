"""Input/output cases for end-to-end ``IndexStore.search()`` tests.

Each ``StoreSearchCase`` supplies the full set of chunks (with
real content so FTS picks them up), optional edges, optional
``changed_files`` for diff proximity, a query, and the expected
property of the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import ChunkKind, Edge, EdgeKind

from tests.index.cases_common import ChunkSpec

@dataclass(frozen=True)
class StoreSearchCase:
    # ── Inputs ───────────────────────────────────────────────────
    chunks: list[ChunkSpec] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    query: str = ""
    top_k: int = 5
    changed_files: frozenset[str] | None = None

    # ── Expected output properties ───────────────────────────────
    expected_count_at_least: int | None = None
    expected_top: str | None = None
    # After the search, assert the given id has a populated breakdown.
    check_breakdown_for_id: str | None = None
    # The id given must rank strictly above every other chunk with
    # the same kind.  Used for importance/proximity ordering.
    expected_first_of_kind: tuple[str, ChunkKind] | None = None
    expected_first_proximity_above_second: bool = False
    expected_first_importance_above_second: bool = False


# ── Ranking: kind boost ─────────────────────────────────────────────


def case_class_outranks_import() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="cls",
                kind=ChunkKind.CLASS,
                name="Engine",
                file_path="src/core.py",
                content="class Engine:\n    pass",
            ),
            ChunkSpec(
                id="imp",
                kind=ChunkKind.IMPORT,
                name="from .core import Engine",
                file_path="src/tools.py",
                content="from .core import Engine",
            ),
        ],
        query="Engine",
        expected_count_at_least=2,
        expected_top="cls",
    )


# ── Ranking: file category penalty ──────────────────────────────────


def case_source_outranks_test_with_44x_tf() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="src",
                name="build_index",
                file_path="src/orchestrator.py",
                content="def build_index(repo): pass",
            ),
            ChunkSpec(
                id="tst",
                name="test_build_index",
                file_path="tests/test_orchestrator.py",
                content="build_index() " * 20,
            ),
        ],
        query="build_index",
        expected_count_at_least=2,
        expected_top="src",
    )


# ── Ranking: exact name match ───────────────────────────────────────


def case_exact_name_ranks_first() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="exact",
                name="_embed_missing",
                file_path="src/orchestrator.py",
                content="def _embed_missing(): pass",
            ),
            ChunkSpec(
                id="partial",
                name="embed_text",
                file_path="src/embeddings.py",
                content=(
                    "def embed_text(text): "
                    "return embed(text) if not missing else None"
                ),
            ),
        ],
        query="_embed_missing",
        expected_count_at_least=1,
        expected_top="exact",
    )


# ── Shape: populated breakdown on a single hit ─────────────────────


def case_single_hit_has_populated_breakdown() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="a",
                name="parse_config",
                content="def parse_config(): pass",
            )
        ],
        query="parse_config",
        expected_count_at_least=1,
        expected_top="a",
        check_breakdown_for_id="a",
    )


# ── Degenerate: no match ────────────────────────────────────────────


def case_gibberish_returns_no_results() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="a",
                name="real_function",
                content="def real_function(): pass",
            )
        ],
        query="zzz_nonexistent_xyz",
        expected_count_at_least=0,
    )


# ── Graceful degradation: no embeddings loaded ─────────────────────


def case_no_embeddings_still_ranks() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="a",
                name="handle_request",
                content="async def handle_request(): pass",
            )
        ],
        query="handle_request",
        expected_count_at_least=1,
        expected_top="a",
    )


# ── Importance: 5 importers → central ranks first ─────────────────


def case_importance_boost_central_above_peripheral() -> StoreSearchCase:
    importers = [
        ChunkSpec(
            id=f"imp{i}",
            kind=ChunkKind.IMPORT,
            name="from config import Config",
            file_path=f"src/mod{i}.py",
            content="from config import Config",
        )
        for i in range(5)
    ]
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="central",
                kind=ChunkKind.CLASS,
                name="Config",
                file_path="src/config.py",
                content="class Config: pass",
            ),
            ChunkSpec(
                id="peripheral",
                kind=ChunkKind.CLASS,
                name="ConfigHelper",
                file_path="src/helper.py",
                content="class ConfigHelper(Config): pass",
            ),
            *importers,
        ],
        edges=[
            Edge(source_id=f"imp{i}", target_id="central", kind=EdgeKind.IMPORTS)
            for i in range(5)
        ],
        query="Config",
        expected_first_of_kind=("central", ChunkKind.CLASS),
        expected_first_importance_above_second=True,
    )


# ── Proximity: changed file ranks first ────────────────────────────


def case_proximity_boost_changed_file() -> StoreSearchCase:
    return StoreSearchCase(
        chunks=[
            ChunkSpec(
                id="changed",
                name="do_work",
                file_path="src/store.py",
                content="def do_work(data): return process(data)",
            ),
            ChunkSpec(
                id="other",
                name="do_work_helper",
                file_path="lib/utils.py",
                content="def do_work_helper(items): return transform(items)",
            ),
        ],
        query="do_work",
        changed_files=frozenset({"src/store.py"}),
        expected_count_at_least=2,
        expected_top="changed",
        expected_first_proximity_above_second=True,
    )

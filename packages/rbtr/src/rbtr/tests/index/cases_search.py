"""Cases for store-level search behaviours.

Each case returns a `SearchScenario` with seed data, query,
and expected results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_cases import case

from rbtr.index.models import ChunkKind, TokenisedChunk

from .conftest import make_chunk


@dataclass(frozen=True)
class SearchScenario:
    """Seed data + query + expected hit IDs."""

    chunks: list[TokenisedChunk]
    query: str
    expected_hit_ids: list[str] = field(default_factory=list)


# ── FTS hits ────────────────────────────────────────────────────────


@case(tags=["fts_hit"])
def case_fts_content_keyword() -> SearchScenario:
    """FTS finds a chunk by a keyword in its content."""
    return SearchScenario(
        chunks=[make_chunk("a", name="greet", content="def greet(): print('hello')")],
        query="hello",
        expected_hit_ids=["a"],
    )


@case(tags=["fts_hit"])
def case_fts_camel_case_split() -> SearchScenario:
    """FTS splits CamelCase and finds by component."""
    return SearchScenario(
        chunks=[make_chunk("a", name="AgentDeps")],
        query="agent",
        expected_hit_ids=["a"],
    )


@case(tags=["fts_hit"])
def case_fts_snake_case_split() -> SearchScenario:
    """FTS splits snake_case and finds by component."""
    return SearchScenario(
        chunks=[make_chunk("a", name="parse_json_response")],
        query="json",
        expected_hit_ids=["a"],
    )


# ── FTS empty ───────────────────────────────────────────────────────


@case(tags=["fts_empty"])
def case_fts_empty_query() -> SearchScenario:
    """Empty query returns no results."""
    return SearchScenario(
        chunks=[make_chunk("a")],
        query="",
    )


@case(tags=["fts_empty"])
def case_fts_nonsense_query() -> SearchScenario:
    """Nonsense query returns no results."""
    return SearchScenario(
        chunks=[make_chunk("a")],
        query="xyzzyplugh",
    )


# ── Name search ─────────────────────────────────────────────────────


@case(tags=["name_hit"])
def case_name_substring_match() -> SearchScenario:
    """Name search finds by substring."""
    return SearchScenario(
        chunks=[make_chunk("a", name="calculate_standard_deviation")],
        query="standard",
        expected_hit_ids=["a"],
    )


@case(tags=["name_empty"])
def case_name_no_match() -> SearchScenario:
    """Name search returns empty for non-matching query."""
    return SearchScenario(
        chunks=[make_chunk("a", name="helper")],
        query="xyzzy",
    )


# ── Unified search (no embeddings) ────────────────────────────


@case(tags=["unified_no_embed"])
def case_unified_search_without_embeddings() -> SearchScenario:
    """Unified search works when no embeddings exist.

    The semantic channel fails gracefully and redistributes
    its weight to FTS + name.
    """
    return SearchScenario(
        chunks=[
            make_chunk("a", name="AppConfig", kind=ChunkKind.CLASS),
            make_chunk("b", name="load_config", kind=ChunkKind.FUNCTION),
        ],
        query="config",
        expected_hit_ids=["a", "b"],
    )

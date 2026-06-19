"""Output DTO projection tests.

The DTOs are the API/wire shape; they drop storage-internal fields
(`id`, `blob_sha`, `embedding`) and omit empty `metadata` so agent
consumers get a clean, low-noise payload.
"""

import json

import pytest

from rbtr.daemon.dto import RefOut, SearchHitOut, SearchSignals, SymbolOut
from rbtr.index.models import Chunk, ChunkKind, ImportMeta, QueryKind, ScoredChunk


@pytest.fixture
def function_chunk() -> Chunk:
    """A function chunk — no import metadata."""
    return Chunk(
        id="abc123",
        blob_sha="deadbeef",
        file_path="src/app.py",
        kind=ChunkKind.FUNCTION,
        name="load_config",
        scope="",
        language="python",
        content="def load_config(): ...",
        line_start=10,
        line_end=20,
    )


@pytest.fixture
def import_chunk() -> Chunk:
    """An import chunk — carries populated import metadata."""
    return Chunk(
        id="def456",
        blob_sha="cafef00d",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name="os",
        language="python",
        content="import os",
        line_start=1,
        line_end=1,
        metadata=ImportMeta(module="os", names="os"),
    )


def test_drops_storage_internals(function_chunk: Chunk) -> None:
    data = json.loads(SymbolOut.from_chunk(function_chunk).model_dump_json())
    assert "id" not in data
    assert "blob_sha" not in data
    assert "embedding" not in data


def test_empty_metadata_is_omitted(function_chunk: Chunk) -> None:
    data = json.loads(SymbolOut.from_chunk(function_chunk).model_dump_json())
    assert "metadata" not in data


def test_populated_metadata_is_kept(import_chunk: Chunk) -> None:
    data = json.loads(SymbolOut.from_chunk(import_chunk).model_dump_json())
    assert data["metadata"] == {
        "module": "os",
        "names": "os",
        "dots": "",
        "language_hint": "",
    }


def test_identity_fields_preserved(function_chunk: Chunk) -> None:
    out = SymbolOut.from_chunk(function_chunk)
    assert (out.name, out.kind, out.file_path, out.line_start, out.line_end) == (
        "load_config",
        ChunkKind.FUNCTION,
        "src/app.py",
        10,
        20,
    )


@pytest.fixture
def scored_chunk() -> ScoredChunk:
    """A workspace search hit with a full signal breakdown."""
    return ScoredChunk(
        id="abc123",
        blob_sha="deadbeef",
        file_path="src/app.py",
        kind=ChunkKind.FUNCTION,
        query_kind=QueryKind.IDENTIFIER,
        name="load_config",
        language="python",
        content="def load_config(): ...",
        line_start=10,
        line_end=20,
        score=0.87,
        lexical=0.5,
        semantic=0.9,
        name_match=0.3,
        kind_boost=1.0,
        file_penalty=1.0,
    )


def test_search_hit_keeps_single_score_only(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk).model_dump_json())
    assert data["score"] == 0.87
    for internal in (
        "lexical",
        "semantic",
        "name_match",
        "kind_boost",
        "file_penalty",
        "importance",
        "proximity",
        "fusion",
        "reranker",
    ):
        assert internal not in data


def test_search_hit_drops_storage_internals(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk).model_dump_json())
    assert "id" not in data
    assert "blob_sha" not in data
    assert "embedding" not in data


def test_search_hit_omits_null_repo_path(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk).model_dump_json())
    assert "repo_path" not in data


def test_search_hit_keeps_repo_path_when_set(scored_chunk: ScoredChunk) -> None:
    hit = SearchHitOut.from_scored(scored_chunk.model_copy(update={"repo_path": "/repos/app"}))
    data = json.loads(hit.model_dump_json())
    assert data["repo_path"] == "/repos/app"


def test_signals_omitted_by_default(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk).model_dump_json())
    assert "signals" not in data


def test_search_hit_omits_empty_match_preview(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk).model_dump_json())
    assert "match_line_offset" not in data
    assert "matched_terms" not in data


def test_search_hit_keeps_match_preview_when_set(scored_chunk: ScoredChunk) -> None:
    hit = SearchHitOut.from_scored(
        scored_chunk.model_copy(
            update={"match_line_offset": 2, "matched_terms": ["config", "load"]}
        )
    )
    data = json.loads(hit.model_dump_json())
    assert data["match_line_offset"] == 2
    assert data["matched_terms"] == ["config", "load"]


def test_ref_out_describes_the_referrer() -> None:
    # Built from an inbound_refs frame row (referrer identity + edge kind).
    row = {
        "name": "os",
        "kind": "import",
        "file_path": "src/app.py",
        "line_start": 1,
        "edge": "imports",
    }
    data = json.loads(RefOut.model_validate(row).model_dump_json())
    assert data == row


def test_signals_included_when_explain(scored_chunk: ScoredChunk) -> None:
    data = json.loads(SearchHitOut.from_scored(scored_chunk, explain=True).model_dump_json())
    assert data["signals"] == {
        "lexical": 0.5,
        "semantic": 0.9,
        "name_match": 0.3,
        "kind_boost": 1.0,
        "file_penalty": 1.0,
        "importance": 1.0,
        "proximity": 1.0,
        "fusion": 0.0,
        "reranker": 0.0,
    }
    assert isinstance(SearchHitOut.from_scored(scored_chunk, explain=True).signals, SearchSignals)

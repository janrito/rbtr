"""Output DTO projection tests.

The DTOs are the API/wire shape; they drop storage-internal fields
(`id`, `blob_sha`, `embedding`) and omit empty `metadata` so agent
consumers get a clean, low-noise payload.
"""

import json

import pytest

from rbtr.config import config
from rbtr.daemon.dto import RefOut, SearchHitOut, SearchSignals, SymbolOut, symbols
from rbtr.domain.models import (
    ChunkKind,
    Definition,
    Import,
    ImportMeta,
    QueryKind,
    ScoredChunk,
    Span,
)
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from ..index.conftest import seed_index


@pytest.fixture
def replies(index: Index) -> dict[ChunkKind, SymbolOut]:
    """A function and an import, written to an index and read back as replies."""
    written = [
        TokenisedChunk(
            blob_sha="deadbeef",
            file_path="src/app.py",
            role=Definition(kind=ChunkKind.FUNCTION, name="load_config"),
            language="python",
            content="def load_config(): ...",
            span=Span(start=10, end=20),
        ),
        TokenisedChunk(
            blob_sha="deadbeef",
            file_path="src/app.py",
            role=Import(name="os", meta=ImportMeta(module="os", names="os")),
            language="python",
            content="import os",
            span=Span(start=1, end=1),
        ),
    ]
    at = index.register("/repo").at("head")
    seed_index(index, written, at)
    return {reply.kind: reply for reply in symbols(index.chunks(at=at))}


def test_drops_storage_internals(replies: dict[ChunkKind, SymbolOut]) -> None:
    data = json.loads(replies[ChunkKind.FUNCTION].model_dump_json())
    assert "id" not in data
    assert "blob_sha" not in data
    assert "embedding" not in data


def test_metadata_is_omitted_on_all_but_imports(replies: dict[ChunkKind, SymbolOut]) -> None:
    data = json.loads(replies[ChunkKind.FUNCTION].model_dump_json())
    assert "metadata" not in data


def test_populated_metadata_is_kept(replies: dict[ChunkKind, SymbolOut]) -> None:
    data = json.loads(replies[ChunkKind.IMPORT].model_dump_json())
    assert data["metadata"] == {
        "module": "os",
        "names": "os",
        "dots": "",
        "language_hint": "",
    }


def test_identity_fields_preserved(replies: dict[ChunkKind, SymbolOut]) -> None:
    out = replies[ChunkKind.FUNCTION]
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
        file_paths=["src/app.py"],
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


@pytest.fixture
def long_scored_chunk(scored_chunk: ScoredChunk) -> ScoredChunk:
    """A hit whose body runs well past the preview cap."""
    body = "\n".join(f"    line_{i} = {i}" for i in range(200))
    return scored_chunk.model_copy(
        update={"content": f"def wide():\n{body}", "line_start": 1, "line_end": 201}
    )


def test_short_hit_previews_whole_body(scored_chunk: ScoredChunk) -> None:
    """A body inside the cap arrives entire, and says so."""
    preview = SearchHitOut.from_scored(scored_chunk).preview
    assert preview.text == scored_chunk.content
    assert preview.clipped is False
    assert preview.total_lines == 1


def test_long_hit_is_clipped_and_reports_its_length(long_scored_chunk: ScoredChunk) -> None:
    """A body past the cap is cut, and carries the length it was cut from."""
    preview = SearchHitOut.from_scored(long_scored_chunk).preview
    assert preview.clipped is True
    assert preview.total_lines == 201
    assert len(preview.text.splitlines()) == config.search_preview_lines
    assert preview.text.startswith("def wide():")


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

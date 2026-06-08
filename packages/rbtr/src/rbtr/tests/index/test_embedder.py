"""Behavioural tests for the embedding model.

Driven by the real (tiny, env-configured) embedding model — real
text in, real vectors out, no mocking of vectors.  Single-threaded by
design: the native model aborts under concurrent access (which is why
the daemon concurrency tests use `stub_embedding_model`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from pytest_mock import MockerFixture

from rbtr.index import _gpu_model
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.frames import ScoredChunkResultRow
from rbtr.index.models import RepoRef
from rbtr.index.store import IndexStore


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder() -> Iterator[Embedder]:
    """The real (tiny) embedding model, loaded once for this module.

    Module-scoped: the native model is loaded and freed once, not per
    test — repeated load/free of the GGUF aborts.  Tests only read
    (embed), never mutate, so sharing one instance is safe.
    """
    instance = Embedder()
    yield instance
    instance.close()


# ── Embedder.embed ───────────────────────────────────────────────────


def test_embed_empty_returns_empty(embedder: Embedder) -> None:
    assert embedder.embed([]) == []


def test_embed_returns_one_vector_per_text(embedder: Embedder) -> None:
    results = embedder.embed(["def add(a, b):\n    return a + b", "def greet(n):\n    return n"])
    assert len(results) == 2
    # Same model → every vector shares the model's width (read, not asserted).
    widths = {len(r.vector) for r in results}
    assert len(widths) == 1
    assert widths.pop() > 0


def test_embed_single_returns_one_vector(embedder: Embedder) -> None:
    vector = embedder.embed_single("def add(a, b):\n    return a + b")
    assert isinstance(vector, list)
    assert len(vector) > 0


def test_embeddings_are_unit_normalised(embedder: Embedder) -> None:
    vector = embedder.embed_single("def add(a, b):\n    return a + b")
    assert math.isclose(math.sqrt(_cosine(vector, vector)), 1.0, abs_tol=1e-3)


def test_embeddings_are_deterministic(embedder: Embedder) -> None:
    text = "def parse(payload):\n    return json.loads(payload)"
    assert _cosine(embedder.embed_single(text), embedder.embed_single(text)) > 0.9999


def test_over_long_input_is_flagged_truncated(mocker: MockerFixture) -> None:
    """Inputs exceeding the context window are marked truncated.

    The one case that needs a controllable double: triggering the flag
    requires a >n_ctx input, and the real model *aborts* when asked to
    embed one.  A stub with a tiny context window exercises the flag
    logic safely.
    """
    model = mocker.MagicMock()
    model.n_ctx.return_value = 32
    model.tokenize.side_effect = lambda b: list(range(len(b) // 4))
    model.embed.side_effect = lambda texts, **_: [[0.1, 0.2, 0.3] for _ in texts]

    embedder = Embedder(model_loader=lambda: model)
    short = "hi"  # 2 bytes → 0 tokens → fits
    long = "x" * 200  # 200 bytes → 50 tokens → exceeds 32
    flags = [result.truncated for result in embedder.embed([short, long])]
    embedder.close()
    assert flags == [False, True]


def test_related_code_outranks_unrelated(embedder: Embedder) -> None:
    """A config-parsing query is nearer parsing code than a greeting."""
    query = embedder.embed_single("read and parse a configuration file")
    parsing = embedder.embed_single(
        "def parse_json(text):\n    import json\n    return json.loads(text)"
    )
    greeting = embedder.embed_single("def greet(name):\n    return f'hello {name}'")
    assert _cosine(query, parsing) > _cosine(query, greeting)


# ── embedding_text ───────────────────────────────────────────────────


def test_embedding_text_prepends_name_to_content() -> None:
    """`embedding_text` is the name, a newline, then the content."""
    content = """\
def save_draft(pr: int) -> None:
    \"\"\"Persist draft to disk.\"\"\"
    pass"""
    assert embedding_text("save_draft", content) == f"save_draft\n{content}"


# ── store.match_by_text ──────────────────────────────────────────────


def test_match_by_text_prepends_query_instruction(mocker: MockerFixture) -> None:
    """The query is prefixed with the model's instruction before embedding.

    The instruction prefix is invisible in the results, so a mock
    embedder is used to observe the text actually embedded — here the
    collaborator call *is* the behaviour under test.
    """
    embedder = mocker.MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    store = IndexStore()
    spy = mocker.patch.object(
        store, "_match_similar", return_value=ScoredChunkResultRow.create_empty()
    )

    store.match_by_text("abc123", "find this", top_k=5, repo_id=1, embedder=embedder)

    embedder.embed_single.assert_called_once_with(
        "Instruct: Given a code search query, retrieve relevant code or documentation"
        "\nQuery:find this"
    )
    spy.assert_called_once_with([RepoRef(repo_id=1, commit_sha="abc123")], [[0.1, 0.2, 0.3]], 5)
    store.close()


# ── resolve_gguf_path validation ─────────────────────────────────────


def test_resolve_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model ID"):
        _gpu_model.resolve_gguf_path("bge-m3")


def test_resolve_two_part_no_gguf_raises(mocker: MockerFixture) -> None:
    mocker.patch.object(_gpu_model, "_list_repo", return_value=["README.md", "model.bin"])
    with pytest.raises(FileNotFoundError, match=r"No \.gguf files"):
        _gpu_model.resolve_gguf_path("org/repo")

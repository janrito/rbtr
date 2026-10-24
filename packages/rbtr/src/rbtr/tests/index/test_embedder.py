"""Behavioural tests for the embedding model.

Driven by the real (tiny, env-configured) embedding model — real
text in, real vectors out, no mocking of vectors.  Single-threaded by
design: the native model aborts under concurrent access (which is why
the daemon concurrency tests use `stub_embedding_model`).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator

import pytest
from pytest_mock import MockerFixture

from rbtr.index import _gpu_model
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.search import _embed_query


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def abort() -> threading.Event:
    """The event `embedder` takes as its *abort*; a test that sets it clears it."""
    return threading.Event()


@pytest.fixture(scope="module")
def embedder(abort: threading.Event) -> Iterator[Embedder]:
    """The real (tiny) embedding model, loaded once for this module.

    Module-scoped: the native model is loaded and freed once, not per
    test — repeated load/free of the GGUF aborts.  Tests only read
    (embed), never mutate, so sharing one instance is safe.
    """
    instance = Embedder(abort=abort)
    yield instance
    instance.close()


def test_embed_empty_returns_empty(embedder: Embedder) -> None:
    assert embedder.embed([]) == []


def test_an_aborted_embed_ends_with_an_error(embedder: Embedder, abort: threading.Event) -> None:
    """llama.cpp checks *abort* while it computes, so a set event ends the call."""
    abort.set()
    try:
        with pytest.raises(RuntimeError):
            embedder.embed(["def helper():\n    return 42\n"])
    finally:
        abort.clear()
    assert len(embedder.embed(["def helper():\n    return 42\n"])) == 1


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


def test_a_batch_past_the_models_limits_embeds_and_flags_what_it_cut(
    embedder: Embedder,
) -> None:
    """A batch longer than one micro-batch, holding a text longer than the
    model's trained context, embeds in full; only that text is truncated."""
    line = "    return sum(v * {n} for v in values if v > {n})  # filter and weigh\n"
    ordinary = [f"def helper_{n}(values):\n" + line.format(n=n) * 4 for n in range(31)]
    long = "def helper(values):\n" + line.format(n=1) * 40
    flags = [result.truncated for result in embedder.embed([*ordinary, long])]
    assert flags == [False] * 31 + [True]


def test_related_code_outranks_unrelated(embedder: Embedder) -> None:
    """A config-parsing query is nearer parsing code than a greeting."""
    query = embedder.embed_single("read and parse a configuration file")
    parsing = embedder.embed_single(
        "def parse_json(text):\n    import json\n    return json.loads(text)"
    )
    greeting = embedder.embed_single("def greet(name):\n    return f'hello {name}'")
    assert _cosine(query, parsing) > _cosine(query, greeting)


def test_embedding_text_prepends_name_to_content() -> None:
    """`embedding_text` is the name, a newline, then the content."""
    content = """\
def save_draft(pr: int) -> None:
    \"\"\"Persist draft to disk.\"\"\"
    pass"""
    assert embedding_text("save_draft", content) == f"save_draft\n{content}"


def test_query_embedding_prepends_instruction(mocker: MockerFixture) -> None:
    """The query is prefixed with the model's instruction before embedding.

    The instruction prefix is invisible in the results, so a mock
    embedder is used to observe the text actually embedded — here the
    collaborator call *is* the behaviour under test.
    """
    prefix = "Instruct: Given a code search query, retrieve relevant code or documentation\nQuery:"
    embedder = mocker.MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    vectors = _embed_query("find this", embedder, None)

    embedder.embed_single.assert_called_once_with(f"{prefix}find this")
    assert vectors == [[0.1, 0.2, 0.3]]


def test_query_embedding_prepends_instruction_to_variants(mocker: MockerFixture) -> None:
    """Expansion variants carry the same prefix as the original query."""
    prefix = "Instruct: Given a code search query, retrieve relevant code or documentation\nQuery:"
    embedder = mocker.MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]
    embedder.embed.return_value = [mocker.MagicMock(vector=[0.4, 0.5, 0.6])]

    vectors = _embed_query("find this", embedder, ["locate that"])

    embedder.embed.assert_called_once_with([f"{prefix}locate that"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_resolve_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model ID"):
        _gpu_model.resolve_gguf_path("bge-m3")


def test_resolve_two_part_no_gguf_raises(mocker: MockerFixture) -> None:
    mocker.patch.object(_gpu_model, "_list_repo", return_value=["README.md", "model.bin"])
    with pytest.raises(FileNotFoundError, match=r"No \.gguf files"):
        _gpu_model.resolve_gguf_path("org/repo")

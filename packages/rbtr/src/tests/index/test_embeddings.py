"""Tests for embedding model loading and encoding.

All tests mock the llama-cpp model to avoid downloading a real GGUF
file.  The mock returns deterministic vectors based on input length.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from llama_cpp import Llama
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.index import embeddings
from rbtr.index.store import IndexStore

type MockLlama = Llama


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None]:
    """Ensure the module-level singleton is clean for each test."""
    embeddings.reset_model()
    yield
    embeddings.reset_model()


def _fake_embed(
    texts: str | list[str],
    normalize: bool = False,
) -> list[list[float]] | list[float]:
    """Deterministic embedding: vector of [len(text)/100] * 4."""
    if isinstance(texts, str):
        n = len(texts) / 100
        return [n, n, n, n]
    return [[len(t) / 100] * 4 for t in texts]


@pytest.fixture
def mock_model(mocker: MockerFixture) -> MockLlama:
    """Fake llama-cpp model with deterministic embed."""
    m = mocker.MagicMock()
    m.embed = mocker.MagicMock(side_effect=_fake_embed)
    return m


# ── embed_texts ──────────────────────────────────────────────────────


def test_embed_texts_empty() -> None:
    result = embeddings.embed_texts([])
    assert result == []


def test_embed_texts_batch(mock_model: MockLlama, mocker: MockerFixture) -> None:

    mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    result = embeddings.embed_texts(["hello", "world!"])
    assert len(result) == 2
    assert len(result[0]) == 4
    assert len(result[1]) == 4
    # Each text is embedded individually to avoid context overflow.
    assert mock_model.embed.call_count == 2  # type: ignore[attr-defined]  # mock attrs on type alias
    mock_model.embed.assert_any_call("hello", normalize=True)  # type: ignore[attr-defined]  # mock attrs on type alias
    mock_model.embed.assert_any_call("world!", normalize=True)  # type: ignore[attr-defined]  # mock attrs on type alias


def test_embed_texts_deterministic(mock_model: MockLlama, mocker: MockerFixture) -> None:

    mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    r1 = embeddings.embed_texts(["abc"])
    embeddings.reset_model()
    r2 = embeddings.embed_texts(["abc"])
    assert r1 == r2


# ── embed_text ───────────────────────────────────────────────────────


def test_embed_text_single(mock_model: MockLlama, mocker: MockerFixture) -> None:

    mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    result = embeddings.embed_text("hello")
    assert isinstance(result, list)
    assert len(result) == 4


# ── get_model / lazy init ────────────────────────────────────────────


def test_get_model_lazy_init(mock_model: MockLlama, mocker: MockerFixture) -> None:

    load = mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    m1 = embeddings.get_model()
    m2 = embeddings.get_model()
    assert m1 is m2
    load.assert_called_once()


def test_reset_model_clears_singleton(mock_model: MockLlama, mocker: MockerFixture) -> None:

    load = mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    embeddings.get_model()
    embeddings.reset_model()
    embeddings.get_model()
    assert load.call_count == 2


def test_reset_model_closes_loaded_model(mock_model: MockLlama, mocker: MockerFixture) -> None:

    mock_model.close = mocker.MagicMock()  # type: ignore[method-assign]  # mock attrs on type alias
    mocker.patch.object(embeddings, "_load_model", return_value=mock_model)

    embeddings.get_model()
    embeddings.reset_model()

    mock_model.close.assert_called_once_with()  # type: ignore[attr-defined]  # mock attrs on type alias


# ── _resolve_model_path ─────────────────────────────────────────────


def test_resolve_explicit_three_part(
    mocker: MockerFixture, config_path: Path, tmp_path: Path
) -> None:

    fake_path = str(tmp_path / "model.gguf")
    config.embedding_model = "org/repo/model.gguf"
    dl = mocker.patch.object(embeddings, "_download", return_value=fake_path)
    path = embeddings._resolve_model_path()
    assert path == Path(fake_path)
    dl.assert_called_once_with("org/repo", "model.gguf", str(config.home / "models"))


def test_resolve_two_part_picks_first_gguf(
    mocker: MockerFixture, config_path: Path, tmp_path: Path
) -> None:

    fake_path = str(tmp_path / "weights.gguf")
    config.embedding_model = "org/repo"
    mocker.patch.object(
        embeddings,
        "_list_repo",
        return_value=["README.md", "weights.gguf", "other.gguf"],
    )
    dl = mocker.patch.object(embeddings, "_download", return_value=fake_path)
    path = embeddings._resolve_model_path()
    assert path == Path(fake_path)
    dl.assert_called_once_with("org/repo", "weights.gguf", str(config.home / "models"))


def test_resolve_invalid_format_raises(config_path: Path) -> None:

    config.embedding_model = "bge-m3"
    with pytest.raises(ValueError, match="Invalid embedding_model"):
        embeddings._resolve_model_path()


def test_resolve_two_part_no_gguf_raises(mocker: MockerFixture, config_path: Path) -> None:

    config.embedding_model = "org/repo"
    mocker.patch.object(
        embeddings,
        "_list_repo",
        return_value=["README.md", "model.bin"],
    )
    with pytest.raises(FileNotFoundError, match=r"No \.gguf files"):
        embeddings._resolve_model_path()


# ── _load_model orchestration ────────────────────────────────────────


def test_load_model_orchestrates_download_and_init(
    mocker: MockerFixture, config_path: Path, tmp_path: Path
) -> None:
    """_load_model resolves path, installs log callback, and constructs Llama."""

    fake_path = tmp_path / "model.gguf"
    fake_path.touch()
    config.embedding_model = "org/repo/model.gguf"

    mocker.patch.object(embeddings, "_resolve_model_path", return_value=fake_path)
    install_cb = mocker.patch.object(embeddings, "_install_llama_log_callback")

    mock_llama_cls = mocker.MagicMock()
    mock_llama_instance = mock_model
    mock_llama_cls.return_value = mock_llama_instance
    mocker.patch("rbtr.index.embeddings.Llama", mock_llama_cls, create=True)
    # Patch the deferred import inside _load_model
    mocker.patch.dict("sys.modules", {"llama_cpp": mocker.MagicMock(Llama=mock_llama_cls)})

    result = embeddings._load_model()

    install_cb.assert_called_once()
    mock_llama_cls.assert_called_once_with(
        model_path=str(fake_path),
        embedding=True,
        n_ctx=0,
        n_gpu_layers=-1,
        verbose=False,
    )
    assert result is mock_llama_instance


# ── Thin wrappers ────────────────────────────────────────────────────


def test_download_delegates_to_hf_hub(mocker: MockerFixture) -> None:
    """_download passes through to hf_hub_download."""
    mock_dl = mocker.MagicMock(return_value="/cached/model.gguf")
    mocker.patch.dict(
        "sys.modules",
        {"huggingface_hub": mocker.MagicMock(hf_hub_download=mock_dl)},
    )
    # Force reimport to pick up the mock
    result = embeddings._download("org/repo", "model.gguf", "/cache")
    assert result == "/cached/model.gguf"
    mock_dl.assert_called_once_with(
        repo_id="org/repo", filename="model.gguf", cache_dir="/cache", token=False
    )


def test_list_repo_delegates_to_hf_hub(mocker: MockerFixture) -> None:
    """_list_repo passes through to list_repo_files."""
    mock_list = mocker.MagicMock(return_value=["a.gguf", "b.txt"])
    mocker.patch.dict(
        "sys.modules",
        {"huggingface_hub": mocker.MagicMock(list_repo_files=mock_list)},
    )
    result = embeddings._list_repo("org/repo")
    assert result == ["a.gguf", "b.txt"]
    mock_list.assert_called_once_with("org/repo")


# ── store.search_by_text integration ────────────────────────────────


def test_search_by_text_embeds_and_searches(
    mocker: MockerFixture,
) -> None:
    mock_embed = mocker.patch(
        "rbtr.index.embeddings.embed_text",
        return_value=[0.1, 0.2, 0.3],
    )

    store = IndexStore()
    spy = mocker.patch.object(store, "search_similar", return_value=[])

    result = store.search_by_text("abc123", "find this", top_k=5)
    assert result == []
    mock_embed.assert_called_once_with("find this")
    spy.assert_called_once_with("abc123", [0.1, 0.2, 0.3], 5, repo_id=1)
    store.close()

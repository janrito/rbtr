"""Tests for the language plugin manager.

Tests the LanguageManager's detection, grammar loading, query
retrieval, and import extraction delegation.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from tree_sitter import Query

from rbtr.config import config
from rbtr.errors import MissingLanguagePluginsError, RbtrError
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction


@pytest.mark.parametrize(
    "path",
    ["README", "data.xyz", "Makefile", "Dockerfile"],
    ids=["no-extension", "unknown-extension", "makefile", "dockerfile"],
)
def test_detect_unknown_returns_none(path: str) -> None:
    assert get_manager().detect_language(path) is None


def test_detect_filename_priority_over_extension(
    monkeypatch: pytest.MonkeyPatch, fresh_manager: None
) -> None:
    """A claimed filename wins over a language that claims the extension."""
    by_name = LanguageRegistration(id="byname", filenames=frozenset({"weird.ext"}))
    by_ext = LanguageRegistration(id="byext", extensions=frozenset({".ext"}))
    eps = [
        SimpleNamespace(name="byname", load=lambda: by_name),
        SimpleNamespace(name="byext", load=lambda: by_ext),
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: eps)
    assert get_manager().detect_language("weird.ext") == "byname"


def test_get_registration_exists() -> None:
    reg = get_manager().get_registration("python")
    assert reg is not None
    assert reg.id == "python"


def test_get_registration_none_for_unknown() -> None:
    assert get_manager().get_registration("nonexistent") is None


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash"],
    ids=["python", "bash"],
)
def test_grammar_base(lang_id: str) -> None:
    assert get_manager().grammar(lang_id) is not None


@pytest.mark.parametrize(
    "lang_id",
    ["nonexistent"],
    ids=["unknown"],
)
def test_grammar_returns_none(lang_id: str) -> None:
    assert get_manager().grammar(lang_id) is None


def test_grammar_cached() -> None:
    """Same object returned on repeated calls."""
    g1 = get_manager().grammar("python")
    g2 = get_manager().grammar("python")
    assert g1 is g2


@pytest.mark.parametrize(
    "lang_id",
    ["svelte", "nonexistent"],
    ids=["chunker-no-query", "unknown"],
)
def test_query_none_for_chunker_or_unknown(lang_id: str) -> None:
    reg = get_manager().get_registration(lang_id)
    assert reg is None or not isinstance(reg.extraction, QueryExtraction)


def test_every_query_compiles_against_its_grammar() -> None:
    """Each registered query compiles — guards against grammar-version drift.

    Query strings reference grammar-specific node types; a grammar bump that
    renames a node makes the whole query fail to compile. Compiling each here
    turns that into one located failure rather than scattered index errors.
    """
    for lang_id in get_manager().all_language_ids():
        reg = get_manager().get_registration(lang_id)
        grammar = get_manager().grammar(lang_id)
        if reg is None or not isinstance(reg.extraction, QueryExtraction) or grammar is None:
            continue
        Query(grammar, reg.extraction.query)  # raises QueryError on an unknown node type


def test_all_language_ids_not_empty() -> None:
    ids = get_manager().all_language_ids()
    assert len(ids) > 0


#
# The manager reads only `ep.name` and `ep.load()`, so each test drives it
# with an inline entry-point double (a `SimpleNamespace`) instead of a real
# `module:attr` EntryPoint. The `fresh_manager` fixture rebuilds the cached
# singleton around the patched discovery and restores it afterwards.


@pytest.fixture
def fresh_manager() -> Iterator[None]:
    get_manager.cache_clear()
    yield
    get_manager.cache_clear()


def test_entrypoint_registration_is_discovered(
    monkeypatch: pytest.MonkeyPatch, fresh_manager: None
) -> None:
    """A `LanguageRegistration` exposed via the entry-point group is registered."""
    reg = LanguageRegistration(id="entrypoint_fake", extensions=frozenset({".epfake"}))
    ep = SimpleNamespace(name="entrypoint_fake", load=lambda: reg)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [ep])
    assert get_manager().detect_language("x.epfake") == "entrypoint_fake"


def test_duplicate_id_raises(monkeypatch: pytest.MonkeyPatch, fresh_manager: None) -> None:
    """Two entry points claiming the same language id is a conflict."""
    reg = LanguageRegistration(id="dup", extensions=frozenset({".dup"}))
    eps = [SimpleNamespace(name="a", load=lambda: reg), SimpleNamespace(name="b", load=lambda: reg)]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: eps)
    with pytest.raises(RbtrError, match="duplicate language id"):
        get_manager()


def test_broken_plugin_is_skipped(monkeypatch: pytest.MonkeyPatch, fresh_manager: None) -> None:
    """An entry point whose module fails to load is skipped, not fatal."""

    def boom() -> LanguageRegistration:
        raise ImportError

    good = LanguageRegistration(id="oklang", extensions=frozenset({".oklang"}))
    eps = [SimpleNamespace(name="broken", load=boom), SimpleNamespace(name="ok", load=lambda: good)]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: eps)
    assert get_manager().all_language_ids() == ["oklang"]


@pytest.fixture
def two_sniffed_languages(monkeypatch: pytest.MonkeyPatch, fresh_manager: None) -> None:
    """A manager that loaded `alpha`, signalled by `%%` lines, and `beta`, by `&&` lines."""
    alpha = LanguageRegistration(id="alpha", content_signals=(re.compile(r"^%%", re.MULTILINE),))
    beta = LanguageRegistration(id="beta", content_signals=(re.compile(r"^&&", re.MULTILINE),))
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [
            SimpleNamespace(name="beta", load=lambda: beta),
            SimpleNamespace(name="alpha", load=lambda: alpha),
        ],
    )


@pytest.mark.parametrize(
    ("text", "language"),
    [
        pytest.param("%% one\n&& two\n&& three\n", "beta", id="most-matches-wins"),
        pytest.param("&& one\n%% two\n", "alpha", id="tie-to-first-id"),
        pytest.param("plain text\n", None, id="no-match"),
    ],
)
def test_a_file_named_by_no_path_takes_the_language_its_content_signals_most(
    two_sniffed_languages: None, text: str, language: str | None
) -> None:
    assert get_manager().detect_language_by_content(text) == language


def test_content_past_the_configured_sample_is_not_read(
    two_sniffed_languages: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "content_sniff_chars", 10)
    assert get_manager().detect_language_by_content("plain text\n%% late\n") is None


#
# The index records which languages it was built with; a rebuild in a
# process missing one of those plugins would re-extract raw chunks and
# churn embeddings.  The manager owns the "what is loaded" knowledge, so
# the check lives here.


@pytest.fixture
def only_oklang(monkeypatch: pytest.MonkeyPatch, fresh_manager: None) -> None:
    """A manager that loaded exactly one language, `oklang`."""
    reg = LanguageRegistration(id="oklang", extensions=frozenset({".oklang"}))
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [SimpleNamespace(name="ok", load=lambda: reg)],
    )


def test_require_languages_loaded_passes_when_all_present(only_oklang: None) -> None:
    get_manager().require_languages_loaded(["oklang"], allow_missing=False)


def test_require_languages_loaded_raises_for_unloaded(only_oklang: None) -> None:
    """A language in the index with no loaded plugin is fatal."""
    with pytest.raises(MissingLanguagePluginsError, match="python"):
        get_manager().require_languages_loaded(["oklang", "python"], allow_missing=False)


def test_require_languages_loaded_allow_missing_downgrades(only_oklang: None) -> None:
    """`allow_missing` overrides the failure (warns instead of raising)."""
    get_manager().require_languages_loaded(["python"], allow_missing=True)


def test_distribution_reports_providing_package() -> None:
    """A bundled language resolves to its `rbtr-lang-*` distribution."""
    dist = get_manager().distribution("python")
    assert dist is not None
    package, version = dist
    assert package == "rbtr-lang-python"
    assert version


def test_distribution_unknown_language_is_none() -> None:
    assert get_manager().distribution("nonesuch") is None


def test_a_block_limited_to_some_languages_holds_those_alone() -> None:
    """Inside the block only the named language is held, Python's `.py` included; after it, the installed languages are back."""
    manager = get_manager()
    named = LanguageRegistration(id="named", extensions=frozenset({".named", ".py"}))
    with manager.limited_to(named):
        assert manager.all_language_ids() == ["named"]
        assert (manager.detect_language("a.named"), manager.detect_language("a.py")) == (
            "named",
            "named",
        )
    assert manager.get_registration("named") is None
    assert (manager.detect_language("a.named"), manager.detect_language("a.py")) == (None, "python")

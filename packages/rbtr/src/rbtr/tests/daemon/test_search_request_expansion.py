"""Validation tests for `SearchRequest`'s keyword / variant normalisation."""

from __future__ import annotations

import pytest

from rbtr.config import config
from rbtr.daemon.messages import SearchRequest


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("keywords", [" load ", "load", "", "config"], ["load", "config"]),  # strip + dedup
        ("variants", ["  a  ", "", "a"], ["a", "a"]),  # strip only, no dedup
    ],
)
def test_keyword_variant_normalisation(field: str, raw: list[str], expected: list[str]) -> None:
    req = SearchRequest.model_validate({"repo_path": "/r", "query": "q", field: raw})
    assert getattr(req, field) == expected


def test_keywords_capped_at_config_max() -> None:
    cap = config.max_expansion_keywords
    req = SearchRequest(repo_path="/r", query="q", keywords=[f"k{i}" for i in range(cap + 5)])
    assert req.keywords is not None
    assert len(req.keywords) == cap


def test_keywords_variants_default_none() -> None:
    req = SearchRequest(repo_path="/r", query="q")
    assert req.keywords is None
    assert req.variants is None

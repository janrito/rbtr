"""Option combinations `rbtr unwatch` and `rbtr forget` refuse."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from rbtr.cli import Forget, Unwatch


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"refs": ["main"], "stale": True}, "says it twice"),
        ({}, "name the refs to stop watching"),
        ({"refs": ["main"], "scope": "all"}, "watched in one repo"),
    ],
)
def test_a_run_drops_one_set_of_refs_chosen_one_way(
    options: Mapping[str, bool | str | list[str]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Unwatch.model_validate(options)


def test_the_repos_that_are_gone_are_not_named_by_path() -> None:
    """`--stale` finds vanished checkouts, so a path cannot select them."""
    with pytest.raises(ValidationError, match="found, not named by path"):
        Forget.model_validate({"stale": True, "repo_path": "/elsewhere"})

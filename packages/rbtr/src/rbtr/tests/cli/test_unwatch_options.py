"""Option combinations `rbtr unwatch` refuses."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from rbtr.cli import Unwatch


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

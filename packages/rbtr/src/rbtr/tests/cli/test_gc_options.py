"""Option combinations `rbtr gc` refuses.

Two retentions leave two different sets of refs behind. They used to
be resolved by precedence, which dropped one without saying so.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from rbtr.cli import Gc


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"orphans": True, "watched_only": True}, "keeps one set of refs"),
        ({"keep_head_only": True, "keep": ["main"]}, "keeps one set of refs"),
        ({"scope": "all", "keep_head_only": True}, "what a single repo keeps"),
        ({"scope": "all", "keep": ["main"]}, "what a single repo keeps"),
    ],
)
def test_one_retention_at_a_time_and_one_every_repo_can_answer(
    options: Mapping[str, bool | list[str]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Gc.model_validate(options)

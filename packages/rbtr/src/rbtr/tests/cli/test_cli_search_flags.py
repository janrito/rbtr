"""The `search` command forwards expansion flags into `SearchRequest`."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.cli import Search
from rbtr.daemon.messages import SearchRequest, SearchResponse


@pytest.fixture
def empty_repo(tmp_path: Path) -> str:
    """A git repo path; no commits are needed to build a request."""
    pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
    return str(tmp_path)


def test_search_forwards_expansion_flags(empty_repo: str, mocker: MockerFixture) -> None:
    captured = mocker.patch("rbtr.cli.try_daemon", return_value=SearchResponse(results=[]))
    command = Search.model_validate(
        {
            "query": "load_config",
            "keywords": ["settings", "configuration"],
            "variants": ["read configuration from file"],
            "repo_path": empty_repo,
        }
    )
    command.cli_cmd()
    request = captured.call_args.args[0]
    assert isinstance(request, SearchRequest)
    assert request.keywords == ["settings", "configuration"]
    assert request.variants == ["read configuration from file"]

"""Read requests that must execute independently of resource waiters."""

from rbtr.daemon.messages import (
    ChangedSymbolsRequest,
    FindRefsRequest,
    ListSymbolsRequest,
    ReadSymbolRequest,
    StatusRequest,
)


def case_status(fake_repo: str) -> StatusRequest:
    return StatusRequest(repo_path=fake_repo)


def case_read_symbol(fake_repo: str) -> ReadSymbolRequest:
    return ReadSymbolRequest(repo_path=fake_repo, symbol="load_config")


def case_list_symbols(fake_repo: str) -> ListSymbolsRequest:
    return ListSymbolsRequest(repo_path=fake_repo, file_path="src/config.py")


def case_find_refs(fake_repo: str) -> FindRefsRequest:
    return FindRefsRequest(repo_path=fake_repo, symbol="load_config")


def case_changed_symbols(
    fake_repo: str, daemon_commit: str, changed_head: str
) -> ChangedSymbolsRequest:
    return ChangedSymbolsRequest(repo_path=fake_repo, base=daemon_commit, head=changed_head)

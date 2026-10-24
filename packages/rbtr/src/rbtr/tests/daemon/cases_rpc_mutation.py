"""Index-changing RPCs overlapping an existing read."""

from rbtr.daemon.messages import ForgetRequest, GcRequest
from rbtr.domain.models import GcMode


def case_forget(fake_repo: str) -> ForgetRequest:
    return ForgetRequest(repo_path=fake_repo)


def case_compact(fake_repo: str) -> GcRequest:
    return GcRequest(repo_path=fake_repo, mode=GcMode.WATCHED, compact=True)

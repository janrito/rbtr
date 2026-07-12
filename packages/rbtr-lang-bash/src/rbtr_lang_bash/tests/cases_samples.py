"""Cases for bash sample extraction tests."""

from __future__ import annotations

from pytest_cases import case

from rbtr.index.models import ChunkKind

type SampleCase = tuple[str, set[ChunkKind]]


@case(id="bash", tags=["sample"])
def case_bash() -> SampleCase:
    """Bash: functions, top-level variable assignments, and source/.
    imports. No classes or methods — all functions are top-level.
    """
    return (
        "bash",
        {ChunkKind.FUNCTION, ChunkKind.VARIABLE, ChunkKind.IMPORT},
    )

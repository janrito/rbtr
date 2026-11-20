"""SQL sample extraction: the `samples/sql/` project, built as rbtr builds a repository and read as an agent reads it.

The snapshots are the golden record of what SQL extraction produces. The
`dialect` tests document how the single generic grammar handles each major
dialect. Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "sql") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached: dict[Target, set[Referrer]] = {}
    with indexed(Path(__file__).parent / "samples" / "sql") as build:
        assert_reached(build, reached)


# rbtr ships one generic SQL grammar for all `.sql` files. This pins current
# extraction per dialect.


@pytest.mark.parametrize(
    "dialect", ["sql_postgres", "sql_mysql", "sql_sqlite", "sql_duckdb", "sql_clickhouse"]
)
def test_sql_dialect_extraction_matches_snapshot(
    indexed: Indexed, dialect: str, snapshot_json: SnapshotAssertion
) -> None:
    """Current extraction for each SQL dialect under the generic grammar."""
    path = f"{dialect}.sql"
    with indexed({path: (Path(__file__).parent / "samples" / path).read_text()}) as build:
        symbols = outline(build)
    assert symbols == snapshot_json

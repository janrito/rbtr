"""`rbtr-eval index` subcommand.

Sequential indexer: for every `<slug>.header.parquet` under
the per-repo dir, build the `full` and `stripped` variants of
that repo into the shared `RBTR_HOME`.  `rbtr index --no-daemon`
blocks until each build is done; no polling, no daemon here.

The sequential loop is what makes it safe to share one home:
only one embedding model ever loads, and DuckDB only sees one
writer at a time.  DVC runs this whole stage as a single
command so the serialisation is visible to the operator.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rbtr.index.models import IndexVariant
from rbtr_eval.rbtr_cli import run_rbtr


class IndexCmd(BaseModel):
    """Build `full` and `stripped` indexes for every per-repo query set."""

    home: Path = Field(description="Shared RBTR_HOME for every variant.")
    repos_dir: Path = Field(description="Directory of cloned repos.")
    per_repo_dir: Path = Field(
        description="Directory of per-repo parquet files; drives which repos to index.",
    )

    def cli_cmd(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)

        # Slug is the filename stem of the header file:
        # `<slug>.header.parquet` → `<slug>`.
        for header_path in sorted(self.per_repo_dir.glob("*.header.parquet")):
            slug = header_path.name.removesuffix(".header.parquet")
            repo_path = (self.repos_dir / slug).resolve()
            if not repo_path.exists():
                msg = f"repo not cloned: {repo_path}"
                raise SystemExit(msg)
            for variant in IndexVariant:
                run_rbtr(
                    [
                        "--home",
                        str(self.home),
                        "index",
                        "--no-daemon",
                        "--variant",
                        variant.value,
                        "--repo-path",
                        str(repo_path),
                    ]
                )

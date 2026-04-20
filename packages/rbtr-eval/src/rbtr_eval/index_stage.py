"""`rbtr-eval index` subcommand.

Sequential indexer: for every `<slug>.jsonl` under the per-repo
dir, build the `full` and `stripped` variants of that repo into
the shared `RBTR_HOME`.  `rbtr index --no-daemon` blocks until
each build is done; no polling, no daemon here.

The sequential loop is what makes it safe to share one home:
only one embedding model ever loads, and DuckDB only sees one
writer at a time.  DVC runs this whole stage as a single
command so the serialisation is visible to the operator.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rbtr.index.models import IndexVariant
from rbtr_eval.extract import load_per_repo
from rbtr_eval.rbtr_cli import run_rbtr


class IndexCmd(BaseModel):
    """Build `full` and `stripped` indexes for every per-repo query set."""

    home: Path = Field(description="Shared RBTR_HOME for every variant.")
    repos_dir: Path = Field(description="Directory of cloned repos.")
    per_repo_dir: Path = Field(
        description="Directory of per-repo JSONL files; drives which repos to index.",
    )

    def cli_cmd(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)

        for jsonl in sorted(self.per_repo_dir.glob("*.jsonl")):
            header, _ = load_per_repo(jsonl)
            repo_path = (self.repos_dir / header.slug).resolve()
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

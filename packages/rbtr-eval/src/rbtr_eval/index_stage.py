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

import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rbtr_eval.extract import load_per_repo

Variant = Literal["full", "stripped"]
_VARIANTS: tuple[Variant, ...] = ("full", "stripped")


def _guard_home(home: Path) -> None:
    real = Path(os.environ.get("RBTR_HOME") or (Path.home() / ".rbtr")).expanduser().resolve()
    requested = home.resolve()
    if requested == real or real.is_relative_to(requested) or requested.is_relative_to(real):
        msg = (
            f"refusing to use --home={requested}: overlaps the user's real "
            f"RBTR_HOME ({real}). Pick a path under data/."
        )
        raise SystemExit(msg)


class IndexCmd(BaseModel):
    """Build `full` and `stripped` indexes for every per-repo query set."""

    home: Path = Field(description="Shared RBTR_HOME for every variant.")
    repos_dir: Path = Field(description="Directory of cloned repos.")
    per_repo_dir: Path = Field(
        description="Directory of per-repo JSONL files; drives which repos to index.",
    )

    def cli_cmd(self) -> None:
        _guard_home(self.home)
        self.home.mkdir(parents=True, exist_ok=True)

        for jsonl in sorted(self.per_repo_dir.glob("*.jsonl")):
            header, _ = load_per_repo(jsonl)
            repo_path = (self.repos_dir / header.slug).resolve()
            if not repo_path.exists():
                msg = f"repo not cloned: {repo_path}"
                raise SystemExit(msg)
            for variant in _VARIANTS:
                subprocess.run(  # noqa: S603 - trusted args
                    [  # noqa: S607 - rbtr on PATH
                        "rbtr",
                        "--home",
                        str(self.home),
                        "index",
                        "--no-daemon",
                        "--variant",
                        variant,
                        "--repo-path",
                        str(repo_path),
                    ],
                    check=True,
                )

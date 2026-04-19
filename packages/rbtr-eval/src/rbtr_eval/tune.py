"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against the dataset's query labels.  Reports the best
triple and the current default for comparison.  Never edits
source — the operator decides whether to copy the suggestion
into `rbtr.index.search._KIND_WEIGHTS`.

Requires `rbtr search --alpha/--beta/--gamma` flags (rbtr
product change, planned for Phase P5 of the rbtr-eval pivot).

Stubbed for Phase P1 — implementation lands in P6.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TuneCmd(BaseModel):
    """Grid-search rbtr's search fusion weights."""

    per_repo_dir: Path = Field(description="Directory holding per-repo JSONL files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    homes_dir: Path = Field(description="Root for per-repo RBTR_HOME directories.")
    grid_step: float = Field(0.2, description="Step size for the (alpha, beta, gamma) grid.")
    output: Path = Field(description="Output path for the tuning suggestion JSON.")

    def cli_cmd(self) -> None:
        msg = "rbtr-eval tune: not implemented yet"
        raise SystemExit(msg)

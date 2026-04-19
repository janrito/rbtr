"""`rbtr-eval measure` subcommand.

Reads `data/dataset.jsonl`, builds two `rbtr` indexes per repo
(default + `--strip-docstrings`) into isolated `RBTR_HOME`
directories under *homes-dir*, replays every query through
`rbtr --json search`, and writes:

* `data/BENCHMARKS.md` — human-readable report.
* `data/metrics.json` — DVC metrics (Hit@1, Hit@3, Hit@10, MRR
  per repo and aggregate).

No imports from `rbtr` — everything is subprocess.

Stubbed for Phase P1 — implementation lands in P4.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MeasureCmd(BaseModel):
    """Build indexes, replay queries, write report + metrics."""

    dataset: Path = Field(description="Path to the merged dataset JSONL.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    homes_dir: Path = Field(description="Root for per-(repo, mode) RBTR_HOME directories.")
    report: Path = Field(description="Output path for BENCHMARKS.md.")
    metrics: Path = Field(description="Output path for metrics JSON.")

    def cli_cmd(self) -> None:
        msg = "rbtr-eval measure: not implemented yet"
        raise SystemExit(msg)

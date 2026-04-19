"""CLI entry point for `rbtr-eval`.

Three subcommands `extract`, `merge-dataset`, `measure`, `tune`
plus the implicit `--help`.  Pydantic-settings drives the
parser, matching the shape used by `rbtr.cli`.

Subcommand bodies live in their own modules — `extract.py`,
`measure.py`, `tune.py`.  The `merge-dataset` subcommand is a
small piece of `extract.py` (same I/O domain).  This module
only wires the parser to those modules' `cli_cmd` methods.
"""

from __future__ import annotations

from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)

from rbtr_eval.extract import ExtractCmd, MergeDatasetCmd
from rbtr_eval.measure import MeasureCmd
from rbtr_eval.tune import TuneCmd


class RbtrEval(
    BaseSettings,
    cli_prog_name="rbtr-eval",
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """rbtr-eval — search-quality evaluation harness."""

    extract: CliSubCommand[ExtractCmd]
    merge_dataset: CliSubCommand[MergeDatasetCmd]
    measure: CliSubCommand[MeasureCmd]
    tune: CliSubCommand[TuneCmd]

    def cli_cmd(self) -> None:
        sub = get_subcommand(self, is_required=False)
        if sub is None:
            CliApp.print_help(self)
            return
        CliApp.run_subcommand(self)


def main() -> None:
    """Entry point for the `rbtr-eval` CLI."""
    cli_source: CliSettingsSource[RbtrEval] = CliSettingsSource(RbtrEval)
    CliApp.run(RbtrEval, cli_settings_source=cli_source)

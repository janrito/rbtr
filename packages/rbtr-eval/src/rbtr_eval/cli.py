"""CLI entry point for `rbtr-eval`.

Five subcommands: `extract`, `index`, `paraphrase`, `measure`,
`tune`, plus the implicit `--help`.  Pydantic-settings drives
the parser, matching the shape used by `rbtr.cli`.

Subcommand bodies live in their own modules.  This module only
wires the parser to those modules' `cli_cmd` methods.  Per-repo
JSONL files are consumed directly by `measure` and `tune`;
there is no merge stage.
"""

from __future__ import annotations

from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)

from rbtr_eval.expand import ExpandCmd
from rbtr_eval.extract import ExtractCmd
from rbtr_eval.index_stage import IndexCmd
from rbtr_eval.measure import MeasureCmd
from rbtr_eval.paraphrase import ParaphraseCmd, ParaphraseReportCmd
from rbtr_eval.profile import ProfileCmd
from rbtr_eval.tune import TuneCmd
from rbtr_eval.tune_reranker import TuneRerankerCmd


class RbtrEval(
    BaseSettings,
    cli_prog_name="rbtr-eval",
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """rbtr-eval — search-quality evaluation harness."""

    expand: CliSubCommand[ExpandCmd]
    extract: CliSubCommand[ExtractCmd]
    index: CliSubCommand[IndexCmd]
    paraphrase: CliSubCommand[ParaphraseCmd]
    paraphrase_report: CliSubCommand[ParaphraseReportCmd]
    profile: CliSubCommand[ProfileCmd]
    measure: CliSubCommand[MeasureCmd]
    tune: CliSubCommand[TuneCmd]
    tune_reranker: CliSubCommand[TuneRerankerCmd]

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

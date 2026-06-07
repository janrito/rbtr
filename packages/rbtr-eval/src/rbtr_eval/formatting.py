"""Shared formatting utilities for eval reports."""

from __future__ import annotations

import polars as pl


# Untyped: accepts any frame shape — used for display
# projections across all reports.
def md_table(df: pl.DataFrame) -> str:
    """Render *df* as a markdown table string via `pl.Config`."""
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        fmt_str_lengths=200,
        tbl_width_chars=10_000,
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        return str(df)


def heading_label(name: str) -> str:
    """Collapse a symbol name to a single line fit for a heading.

    A multi-line name (e.g. a CSS selector group) would break a
    markdown `###` heading: only the first line renders as the
    heading and the rest leaks into body text.  Return the first
    non-empty line, suffixed with ` …` when more lines followed,
    so the heading stays valid.  The full name should still be
    shown verbatim in the body.
    """
    lines = [line.strip() for line in name.splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return name.strip()
    head = nonempty[0]
    return f"{head} …" if len(nonempty) > 1 else head

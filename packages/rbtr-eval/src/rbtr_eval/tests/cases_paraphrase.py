"""Cases for end-to-end `paraphrase_symbols` runs.

Each case returns the query frame targeting one `lib.py` chunk of the
`mixed_kind_repo` corpus.

The anonymous chunk is the case that earns its place: `comment` and
`raw_chunk` carry no name, and the empty string is a substring of every
paraphrase, so an exclusion list admitting it rejects them all.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
from pytest_cases import case

from rbtr_eval.schemas import QueryRow


@case(tags=["named"])
def case_named_function() -> dy.DataFrame[QueryRow]:
    """`connect` — the concept row carries the name back."""
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "lib.py",
                "scope": "",
                "name": "connect",
                "line_start": 7,
                "line_end": 9,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "body",
                "text": "seed",
            }
        ]
    ).pipe(QueryRow.validate, cast=True)


def case_anonymous_comment() -> dy.DataFrame[QueryRow]:
    """The standalone comment in `lib.py`, which carries no name."""
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "lib.py",
                "scope": "",
                "name": "",
                "line_start": 5,
                "line_end": 5,
                "symbol_kind": "comment",
                "language": "python",
                "provenance": "body",
                "text": "seed",
            }
        ]
    ).pipe(QueryRow.validate, cast=True)

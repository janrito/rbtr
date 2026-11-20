"""Top-level statements, declared and not: a file of the test language `code` and every symbol `list-symbols` gives for it.

`code` is Python's grammar under a query that declares a module's `if` as
a `@statement`, captures functions, comments and calls, and nothing else,
so what is not captured is left to fragments. Each symbol is
`(kind, names, line_start, line_end)`.
"""

from __future__ import annotations

from dataclasses import dataclass

type Symbol = tuple[str, str, int, int]


@dataclass(frozen=True)
class StatementScenario:
    """A file's text and every symbol it gives."""

    text: str
    symbols: list[Symbol]


def case_a_declared_statement_holding_nothing_captured_is_one_chunk() -> StatementScenario:
    return StatementScenario(
        text="""\
if ready:
    go()
""",
        symbols=[("statement", "", 1, 2)],
    )


def case_a_definition_in_a_declared_statement_is_a_chunk_nested_in_it() -> StatementScenario:
    return StatementScenario(
        text="""\
if __name__ == "__main__":
    def main():
        pass
    main()
""",
        symbols=[("statement", "", 1, 4), ("function", "main", 2, 3)],
    )


def case_a_comment_flush_above_a_declared_statement_is_in_it() -> StatementScenario:
    return StatementScenario(
        text="""\
# Run when called as a script.
if __name__ == "__main__":
    go()
""",
        symbols=[("statement", "", 1, 3)],
    )


def case_an_undeclared_statement_holding_a_capture_leaves_text_around_it() -> StatementScenario:
    """A `while` no pattern declares, holding a `def`: the text before the `def` is one fragment, the text after another."""
    return StatementScenario(
        text="""\
while running:
    def step():
        pass
    step()
""",
        symbols=[
            ("fragment", "", 1, 1),
            ("function", "step", 2, 3),
            ("fragment", "", 4, 4),
        ],
    )

"""Bash extraction test cases.

Each `@case` returns test data consumed by `test_extraction.py` via
`pytest-cases`.
"""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]
type MixedCase = tuple[dict[str, str], set[str], list[tuple[str, str]]]


@case(tags=["symbol"])
def case_bash_function_keyword() -> SymbolCase:
    """function deploy { ... }."""
    src = """\
function deploy {
    echo deploying
}
"""
    return {"input.sh": src}, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_function_keyword_parens() -> SymbolCase:
    """function deploy() { ... }."""
    src = """\
function deploy() {
    echo deploying
}
"""
    return {"input.sh": src}, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_function_posix() -> SymbolCase:
    """deploy() { ... } — POSIX syntax."""
    src = """\
deploy() {
    echo deploying
}
"""
    return {"input.sh": src}, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_multiple_functions() -> SymbolCase:
    """Multiple shell functions."""
    src = """\
function setup {
    echo setup
}

function teardown {
    echo teardown
}

run() {
    echo run
}
"""
    return (
        {"input.sh": src},
        [
            ("function", "setup", ""),
            ("function", "teardown", ""),
            ("function", "run", ""),
        ],
    )


@case(tags=["symbol"])
def case_bash_alias() -> SymbolCase:
    """An alias is a variable, named without the `=` the grammar fuses on."""
    src = """\
alias ll="ls -l"
"""
    return {"input.sh": src}, [("variable", "ll", "")]


@case(tags=["symbol"])
def case_bash_function_local_vars() -> SymbolCase:
    """Function with local variables."""
    src = """\
setup() {
    local dir="/tmp"
    mkdir -p "$dir"
}
"""
    return {"input.sh": src}, [("function", "setup", "")]


@case(tags=["symbol"])
def case_bash_function_conditionals() -> SymbolCase:
    """Function with conditionals."""
    src = """\
check() {
    if [ -f /tmp/x ]; then
        echo yes
    fi
}
"""
    return {"input.sh": src}, [("function", "check", "")]


@case(tags=["symbol"])
def case_bash_assignment() -> SymbolCase:
    """Top-level variable assignment."""
    return {"input.sh": "MAX=100\n"}, [("variable", "MAX", "")]


@case(tags=["mixed"])
def case_bash_full_script() -> MixedCase:
    """Realistic shell script with doc comments on every
    function.  Expected-kinds tuple pins symbol extraction;
    content invariants are in `test_docstrings.py`.
    """
    src = """\
#!/bin/bash

# Deploy the current build to the given environment.
deploy() {
    local env="$1"
    echo "deploying to $env"
}

# Roll back the last deploy.
rollback() {
    echo "rolling back"
}
"""
    return {"input.sh": src}, {"function"}, []

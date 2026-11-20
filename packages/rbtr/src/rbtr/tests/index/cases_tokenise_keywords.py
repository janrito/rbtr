"""Cases for the FTS-roundtrip keyword test.

Each case returns a `KeywordScenario`: one source file, the keyword
its function's body holds, and the name of that function.  The
fixture in `test_tokenisation.py` builds the file into an index.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordScenario:
    keyword: str
    path: str
    name: str
    content: str


def case_python_import() -> KeywordScenario:
    return KeywordScenario(
        keyword="import",
        path="src/py_import.py",
        name="from pathlib import Path",
        content="""\
from pathlib import Path
""",
    )


def case_python_for() -> KeywordScenario:
    return KeywordScenario(
        keyword="for",
        path="src/py_for.py",
        name="process_items",
        content="""\
def process_items(collection):
    for item in collection:
        yield item
""",
    )


def case_python_if() -> KeywordScenario:
    return KeywordScenario(
        keyword="if",
        path="src/py_if.py",
        name="check_value",
        content="""\
def check_value(value):
    if value is not None:
        return value
""",
    )


def case_python_with() -> KeywordScenario:
    return KeywordScenario(
        keyword="with",
        path="src/py_with.py",
        name="read_file",
        content="""\
def read_file(path):
    with open(path) as f:
        return f.read()
""",
    )


def case_python_async() -> KeywordScenario:
    return KeywordScenario(
        keyword="async",
        path="src/py_async.py",
        name="fetch_data",
        content="""\
async def fetch_data(url):
    return await get(url)
""",
    )


def case_sql_select() -> KeywordScenario:
    return KeywordScenario(
        keyword="select",
        path="src/sql_select.py",
        name="query_users",
        content="""\
def query_users():
    return "SELECT id, name FROM users WHERE active"
""",
    )


def case_sql_into() -> KeywordScenario:
    return KeywordScenario(
        keyword="into",
        path="src/sql_insert.py",
        name="save_record",
        content="""\
def save_record():
    return "INSERT INTO chunks (id, content) VALUES (?, ?)"
""",
    )


def case_rust_impl() -> KeywordScenario:
    return KeywordScenario(
        keyword="impl",
        path="src/rs_impl.py",
        name="display_impl",
        content="""\
def display_impl():
    return "fn show() -> impl Display"
""",
    )


def case_go_defer() -> KeywordScenario:
    return KeywordScenario(
        keyword="defer",
        path="src/go_defer.py",
        name="close_file",
        content="""\
def close_file():
    return "defer file.Close()"
""",
    )


def case_js_this() -> KeywordScenario:
    return KeywordScenario(
        keyword="this",
        path="src/js_this.js",
        name="set_state",
        content="""\
function set_state() {
  this.setState({loading: true});
}
""",
    )

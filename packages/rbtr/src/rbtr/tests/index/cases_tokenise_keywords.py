"""Cases for the FTS-roundtrip keyword test.

Each case returns a `KeywordScenario` describing what keyword to
search for and what chunk should match it.  The case holds only
data; the fixture in `test_tokenise_keywords.py` builds the
full `Chunk` including tokenisation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordScenario:
    keyword: str
    chunk_id: str
    name: str
    content: str


def case_python_import() -> KeywordScenario:
    return KeywordScenario(
        keyword="import",
        chunk_id="py_import",
        name="load_modules",
        content="from pathlib import Path",
    )


def case_python_for() -> KeywordScenario:
    return KeywordScenario(
        keyword="for",
        chunk_id="py_for",
        name="process_items",
        content="for item in collection: yield item",
    )


def case_python_if() -> KeywordScenario:
    return KeywordScenario(
        keyword="if",
        chunk_id="py_if",
        name="check_value",
        content="if value is not None: return value",
    )


def case_python_with() -> KeywordScenario:
    return KeywordScenario(
        keyword="with",
        chunk_id="py_with",
        name="read_file",
        content="with open(path) as f: data = f.read()",
    )


def case_python_async() -> KeywordScenario:
    return KeywordScenario(
        keyword="async",
        chunk_id="py_async",
        name="fetch_data",
        content="async def fetch_data(url): return await get(url)",
    )


def case_sql_select() -> KeywordScenario:
    return KeywordScenario(
        keyword="select",
        chunk_id="sql_select",
        name="query_users",
        content="SELECT id, name FROM users WHERE active",
    )


def case_sql_into() -> KeywordScenario:
    return KeywordScenario(
        keyword="into",
        chunk_id="sql_insert",
        name="save_record",
        content="INSERT INTO chunks (id, content) VALUES (?, ?)",
    )


def case_rust_impl() -> KeywordScenario:
    return KeywordScenario(
        keyword="impl",
        chunk_id="rs_impl",
        name="display_impl",
        content="impl Display for Config { fn fmt() }",
    )


def case_go_defer() -> KeywordScenario:
    return KeywordScenario(
        keyword="defer",
        chunk_id="go_defer",
        name="close_file",
        content="defer file.Close()",
    )


def case_js_this() -> KeywordScenario:
    return KeywordScenario(
        keyword="this",
        chunk_id="js_this",
        name="set_state",
        content="this.setState({loading: true})",
    )

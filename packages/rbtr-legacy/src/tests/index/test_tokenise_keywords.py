"""Tests that language keywords survive tokenisation and FTS indexing.

With the old `stopwords='english'` setting, Python keywords like
`if`, `for`, `import`, `in`, `is`, `not`, `or`, `and`,
`as`, `with` were silently removed by DuckDB's FTS tokeniser.
SQL keywords like `SELECT`, `INSERT INTO`, `WHERE` were even
worse — most are English stopwords.

These tests verify that:
1. `tokenise_code` preserves all language keywords.
2. The FTS index with `stopwords='none'` can find them.
"""

from __future__ import annotations

import pytest

from rbtr_legacy.index.models import Chunk, ChunkKind
from rbtr_legacy.index.store import IndexStore
from rbtr_legacy.index.tokenise import tokenise_code

# ── Tokenisation: keywords survive ───────────────────────────────────
#
# Each case is (code_snippet, keywords_that_must_appear_in_tokens).
# Grouped by language.


@pytest.mark.parametrize(
    ("code", "expected_keywords"),
    [
        # ── Python ───────────────────────────────────────────────
        ("async def fetch(url): pass", {"async", "def"}),
        ("await response.json()", {"await"}),
        ("with open(path) as f:", {"with", "as"}),
        ("for item in collection:", {"for", "in"}),
        ("if x and not y or z:", {"if", "and", "not", "or"}),
        ("from pathlib import Path", {"from", "import"}),
        ("if value is not None:", {"is", "not"}),
        ("class Handler(Base):", {"class"}),
        ("yield from generator", {"yield", "from"}),
        ("raise ValueError(msg)", {"raise"}),
        ("try:\n    pass\nexcept Exception as e:", {"try", "except", "as"}),
        ("while running:", {"while"}),
        ("return result", {"return"}),
        ("elif condition:", {"elif"}),
        ("assert x == y", {"assert"}),
        ("global counter", {"global"}),
        ("del items[0]", {"del"}),
        ("lambda x: x + 1", {"lambda"}),
        # ── JavaScript / TypeScript ──────────────────────────────
        ("const data = await fetch(url)", {"const", "await"}),
        ("for (const item of items) {", {"for", "of", "const"}),
        ("this.setState({loading: true})", {"this"}),
        ("export default function App() {", {"export", "default", "function"}),
        ("if (condition) { return; }", {"if", "return"}),
        ("let x = new Map()", {"let", "new"}),
        ("typeof value === 'string'", {"typeof", "value"}),
        ("import { useState } from 'react'", {"import", "from"}),
        ("switch (action) { case 'add': break; }", {"switch", "case", "break"}),
        ("try { } catch (e) { throw e; }", {"try", "catch", "throw"}),
        # ── Go ───────────────────────────────────────────────────
        ("if err != nil { return err }", {"if", "err", "nil", "return"}),
        ("for i, v := range items {", {"for", "range"}),
        ("func (s *Server) handle()", {"func"}),
        ("switch v := value.(type) {", {"switch", "type"}),
        ("go handleConnection(conn)", {"go"}),
        ("defer file.Close()", {"defer"}),
        ("select { case msg := <-ch: }", {"select", "case"}),
        ("chan int", {"chan", "int"}),
        ("interface{}", {"interface"}),
        # ── Rust ─────────────────────────────────────────────────
        ("impl Display for MyStruct {", {"impl", "for"}),
        ("if let Some(v) = option {", {"if", "let"}),
        ("match result { Ok(v) => v }", {"match"}),
        ("pub fn process(data: &[u8]) -> Result<()>", {"pub", "fn"}),
        ("use std::collections::HashMap", {"use", "std"}),
        ("async fn fetch() -> Result<()>", {"async", "fn"}),
        ("mod tests { }", {"mod"}),
        ("trait Serialize { }", {"trait"}),
        ("enum Color { Red, Green }", {"enum"}),
        ("unsafe { ptr::read(src) }", {"unsafe", "ptr"}),
        ("loop { break; }", {"loop", "break"}),
        ("where T: Clone + Send", {"where"}),
        # ── Java ─────────────────────────────────────────────────
        ("this.name = name;", {"this", "name"}),
        ("public static void main(String[] args)", {"public", "static", "void", "main"}),
        ("import java.util.List;", {"import", "java", "util"}),
        ("synchronized (lock) { }", {"synchronized", "lock"}),
        ("throw new RuntimeException(msg)", {"throw", "new"}),
        ("final int MAX = 100;", {"final", "int"}),
        ("instanceof String", {"instanceof"}),
        ("extends AbstractClass implements Interface", {"extends", "implements"}),
        # ── C / C++ ──────────────────────────────────────────────
        ("if (not found && ptr != nullptr) {", {"if", "not", "ptr", "nullptr"}),
        ("#include <stdio.h>", {"include", "stdio", "h"}),
        ("typedef struct node_t { } Node;", {"typedef", "struct", "node_t"}),
        ("static inline void process()", {"static", "inline", "void", "process"}),
        ("template<typename T> class Vec { }", {"template", "typename", "class"}),
        ("namespace fs = std::filesystem;", {"namespace", "fs", "std", "filesystem"}),
        ("virtual void draw() override = 0", {"virtual", "void", "draw", "override"}),
        ("const auto& ref = value;", {"const", "auto", "ref", "value"}),
        # ── Ruby ─────────────────────────────────────────────────
        ("if condition then do_something end", {"if", "then", "end"}),
        ("unless error.nil?", {"unless"}),
        ("def initialize(name)", {"def", "initialize"}),
        ("module ActiveSupport", {"module"}),
        ("require 'json'", {"require", "json"}),
        ("attr_accessor :name", {"attr_accessor"}),
        ("begin rescue ensure end", {"begin", "rescue", "ensure", "end"}),
        # ── SQL ──────────────────────────────────────────────────
        ("SELECT id, name FROM users", {"select", "id", "name", "from", "users"}),
        ("INSERT INTO chunks (id) VALUES (?)", {"insert", "into", "chunks", "id", "values"}),
        (
            "WHERE active = true AND role IN ('admin')",
            {"where", "active", "true", "and", "role", "in", "admin"},
        ),
        ("ORDER BY created_at DESC LIMIT 10", {"order", "by", "desc", "limit"}),
        ("LEFT JOIN edges ON source_id = id", {"left", "join", "on"}),
        ("GROUP BY kind HAVING count > 1", {"group", "by", "having", "count"}),
        ("CREATE TABLE IF NOT EXISTS meta", {"create", "table", "if", "not", "exists", "meta"}),
        # ── Bash ─────────────────────────────────────────────────
        ("if [ -f file ]; then echo found; fi", {"if", "then", "echo", "fi"}),
        ("for f in *.py; do echo $f; done", {"for", "f", "in", "do", "echo", "done"}),
        ("while read line; do process; done", {"while", "read", "line", "do", "done"}),
        ("case $1 in start) ;; esac", {"case", "in", "start", "esac"}),
    ],
    ids=lambda val: str(sorted(val) if isinstance(val, set) else val)[:50],
)
def test_keywords_survive_tokenisation(code: str, expected_keywords: set[str]) -> None:
    """Language keywords are preserved by tokenise_code."""
    tokens = tokenise_code(code).split()
    for kw in expected_keywords:
        assert kw in tokens, f"keyword {kw!r} missing from tokens of {code!r}"


# ── FTS roundtrip: keywords are searchable ───────────────────────────
#
# Insert chunks containing keyword-heavy code, verify that searching
# for those keywords finds them.  This catches regressions in FTS
# settings (e.g. accidentally re-enabling stopwords).


def _make_chunk(chunk_id: str, name: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        blob_sha=f"blob_{chunk_id}",
        file_path=f"src/{chunk_id}.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
    )


# Keywords that were killed by English stopwords in the old FTS config.
# Each tuple: (keyword, chunk with that keyword in content)
_KEYWORD_CHUNKS: list[tuple[str, Chunk]] = [
    (
        "import",
        _make_chunk("py_import", "load_modules", "from pathlib import Path"),
    ),
    (
        "for",
        _make_chunk("py_for", "process_items", "for item in collection: yield item"),
    ),
    (
        "if",
        _make_chunk("py_if", "check_value", "if value is not None: return value"),
    ),
    (
        "with",
        _make_chunk("py_with", "read_file", "with open(path) as f: data = f.read()"),
    ),
    (
        "async",
        _make_chunk("py_async", "fetch_data", "async def fetch_data(url): return await get(url)"),
    ),
    (
        "select",
        _make_chunk("sql_select", "query_users", "SELECT id, name FROM users WHERE active"),
    ),
    (
        "into",
        _make_chunk("sql_insert", "save_record", "INSERT INTO chunks (id, content) VALUES (?, ?)"),
    ),
    (
        "impl",
        _make_chunk("rs_impl", "display_impl", "impl Display for Config { fn fmt() }"),
    ),
    (
        "defer",
        _make_chunk("go_defer", "close_file", "defer file.Close()"),
    ),
    (
        "this",
        _make_chunk("js_this", "set_state", "this.setState({loading: true})"),
    ),
]


@pytest.fixture
def keyword_store() -> IndexStore:
    """Store seeded with keyword-containing chunks."""
    store = IndexStore()
    chunks = [chunk for _, chunk in _KEYWORD_CHUNKS]
    store.insert_chunks(chunks)
    for c in chunks:
        store.insert_snapshot("head", c.file_path, c.blob_sha)
    store.rebuild_fts_index()
    return store


@pytest.mark.parametrize(
    ("keyword", "expected_id"),
    [(kw, chunk.id) for kw, chunk in _KEYWORD_CHUNKS],
    ids=[kw for kw, _ in _KEYWORD_CHUNKS],
)
def test_keyword_searchable_via_fts(
    keyword_store: IndexStore,
    keyword: str,
    expected_id: str,
) -> None:
    """Keywords that were English stopwords are now findable via BM25."""
    results = keyword_store.search_fulltext("head", keyword, top_k=10)
    ids = [c.id for c, _ in results]
    assert expected_id in ids, f"searching for {keyword!r} did not find {expected_id!r}, got {ids}"

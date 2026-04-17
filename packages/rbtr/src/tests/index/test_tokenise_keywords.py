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
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from tests.index.case_tokenise_keywords import KeywordCase

# ── Tokenisation: keywords survive ───────────────────────────────────


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
        ("func handle(w http.ResponseWriter, r *http.Request) {", {"func", "handle"}),
        ("defer file.Close()", {"defer"}),
        ("go process(item)", {"go", "process"}),
        ("select { case x := <-ch: }", {"select", "case"}),
        ("chan int", {"chan"}),
        ("package main", {"package", "main"}),
        ("struct { name string }", {"struct", "name", "string"}),
        ("interface { Read() }", {"interface"}),
        # ── Rust ─────────────────────────────────────────────────
        ("impl Display for Config {", {"impl", "for"}),
        ("let mut data = Vec::new();", {"let", "mut"}),
        ("fn process(item: &str) -> Result<T, E> {", {"fn"}),
        ("pub fn main() {", {"pub", "fn"}),
        ("match value { Some(x) => x, None => 0 }", {"match"}),
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


# ── FTS roundtrip: keywords are searchable ──────────────────────────


@fixture
@parametrize_with_cases(
    "case",
    cases="tests.index.case_tokenise_keywords",
    prefix="case_",
)
def keyword_store_and_case(case: KeywordCase) -> tuple[IndexStore, KeywordCase]:
    """Seed a fresh store with the single chunk the case describes."""
    chunk = Chunk(
        id=case.chunk_id,
        blob_sha=f"blob_{case.chunk_id}",
        file_path=f"src/{case.chunk_id}.py",
        kind=ChunkKind.FUNCTION,
        name=case.name,
        content=case.content,
        content_tokens=tokenise_code(case.content),
        name_tokens=tokenise_code(case.name),
        line_start=1,
        line_end=1,
    )
    store = IndexStore()
    store.insert_chunks([chunk])
    store.insert_snapshot("head", chunk.file_path, chunk.blob_sha)
    store.rebuild_fts_index()
    return store, case


def test_keyword_searchable_via_fts(
    keyword_store_and_case: tuple[IndexStore, KeywordCase],
) -> None:
    """Keywords that were English stopwords are now findable via BM25."""
    store, case = keyword_store_and_case
    results = store.search_fulltext("head", case.keyword, top_k=10)
    ids = [c.id for c, _ in results]
    assert case.chunk_id in ids, (
        f"searching for {case.keyword!r} did not find {case.chunk_id!r}, "
        f"got {ids}"
    )

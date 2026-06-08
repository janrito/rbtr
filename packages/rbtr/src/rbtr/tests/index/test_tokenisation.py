"""Tests for code-aware tokenisation."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import Snapshot
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

from .cases_tokenise_keywords import KeywordScenario
from .conftest import make_chunk


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ── Core splitting rules ─────────────────────────────────
        ("AgentDeps", "agentdeps agent deps"),
        ("getUserById", "getuserbyid get user by id"),
        ("_deep_merge", "_deep_merge deep merge"),
        ("HTTP_STATUS_CODE", "http_status_code http status code"),
        ("x", "x"),
        ("XMLParser", "xmlparser xml parser"),
        ("parseHTML5Doc", "parsehtml5doc parse html 5 doc"),
        ("__init__", "__init__ init"),
        ("list[float]", "list float"),
        ("", ""),
        # Repeated identifiers should be deduplicated.
        ("foo foo foo", "foo"),
        # Pure punctuation / operators produce nothing.
        ("= + - / *", ""),
        # Mixed code line.
        ("def build_index(repo, ref):", "def build_index build index repo ref"),
        # ── Python ───────────────────────────────────────────────
        ("StatisticsCalculator", "statisticscalculator statistics calculator"),
        ("MAX_RETRY_COUNT", "max_retry_count max retry count"),
        ("list[dict[str, Any]]", "list dict str any"),
        ("ModelMessage", "modelmessage model message"),
        ("RunContext", "runcontext run context"),
        ("ThinkingEffort", "thinkingeffort thinking effort"),
        ("IndexStore", "indexstore index store"),
        ("build_model", "build_model build model"),
        # ── Java / C# ───────────────────────────────────────────
        ("AbstractHttpRequestHandler", "abstracthttprequesthandler abstract http request handler"),
        ("getResponseBody", "getresponsebody get response body"),
        ("DEFAULT_BUFFER_SIZE", "default_buffer_size default buffer size"),
        ("HashMap<String, List<Integer>>", "hashmap hash map string list integer"),
        # ── Go ───────────────────────────────────────────────────
        ("HandleHTTPRequest", "handlehttprequest handle http request"),
        ("parseConfig", "parseconfig parse config"),
        ("ReadWriteCloser", "readwritecloser read write closer"),
        ("ErrNotFound", "errnotfound err not found"),
        # ── Rust ─────────────────────────────────────────────────
        ("process_http_request", "process_http_request process http request"),
        ("HttpResponseBuilder", "httpresponsebuilder http response builder"),
        ("ConnectionRefused", "connectionrefused connection refused"),
        # ── C / C++ ──────────────────────────────────────────────
        ("ARRAY_SIZE", "array_size array size"),
        ("XMLHttpRequest", "xmlhttprequest xml http request"),
        ("getNodeValue", "getnodevalue get node value"),
        ("std::vector<int>", "std vector int"),
        ("uint32_t", "uint32_t uint 32 t"),
        # ── JavaScript / TypeScript ──────────────────────────────
        ("getElementById", "getelementbyid get element by id"),
        ("UserProfileCard", "userprofilecard user profile card"),
        ("MAX_CONNECTIONS", "max_connections max connections"),
        ("IEventEmitter", "ieventemitter i event emitter"),
        ("Promise<Response>", "promise response"),
        # ── Ruby ─────────────────────────────────────────────────
        ("find_or_create_by", "find_or_create_by find or create by"),
        ("ActiveRecord", "activerecord active record"),
        ("RAILS_ENV", "rails_env rails env"),
        # ── Bash ─────────────────────────────────────────────────
        ("BASH_SOURCE", "bash_source bash source"),
        ("check_dependencies", "check_dependencies check dependencies"),
    ],
    ids=lambda val: repr(val)[:40],
)
def test_tokenise_code(text: str, expected: str) -> None:
    assert tokenise_code(text) == expected


# ── Keywords survive FTS indexing ────────────────────────────────────
#
# Language keywords must not be stripped by FTS stopwords.
# See case_tokenise_keywords.py for the keyword scenarios.


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
        # ── Go ──────────────────────────────────────────────────
        ("func handle(w http.ResponseWriter, r *http.Request) {", {"func", "handle"}),
        ("defer file.Close()", {"defer"}),
        ("go process(item)", {"go", "process"}),
        ("select { case x := <-ch: }", {"select", "case"}),
        ("chan int", {"chan"}),
        ("package main", {"package", "main"}),
        ("struct { name string }", {"struct", "name", "string"}),
        ("interface { Read() }", {"interface"}),
        # ── Rust ────────────────────────────────────────────────
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
        # ── Java ────────────────────────────────────────────────
        ("this.name = name;", {"this", "name"}),
        ("public static void main(String[] args)", {"public", "static", "void", "main"}),
        ("import java.util.List;", {"import", "java", "util"}),
        ("synchronized (lock) { }", {"synchronized", "lock"}),
        ("throw new RuntimeException(msg)", {"throw", "new"}),
        ("final int MAX = 100;", {"final", "int"}),
        ("instanceof String", {"instanceof"}),
        ("extends AbstractClass implements Interface", {"extends", "implements"}),
        # ── C / C++ ─────────────────────────────────────────────
        ("if (not found && ptr != nullptr) {", {"if", "not", "ptr", "nullptr"}),
        ("#include <stdio.h>", {"include", "stdio", "h"}),
        ("typedef struct node_t { } Node;", {"typedef", "struct", "node_t"}),
        ("static inline void process()", {"static", "inline", "void", "process"}),
        ("template<typename T> class Vec { }", {"template", "typename", "class"}),
        ("namespace fs = std::filesystem;", {"namespace", "fs", "std", "filesystem"}),
        ("virtual void draw() override = 0", {"virtual", "void", "draw", "override"}),
        ("const auto& ref = value;", {"const", "auto", "ref", "value"}),
        # ── Ruby ────────────────────────────────────────────────
        ("if condition then do_something end", {"if", "then", "end"}),
        ("unless error.nil?", {"unless"}),
        ("def initialize(name)", {"def", "initialize"}),
        ("module ActiveSupport", {"module"}),
        ("require 'json'", {"require", "json"}),
        ("attr_accessor :name", {"attr_accessor"}),
        ("begin rescue ensure end", {"begin", "rescue", "ensure", "end"}),
        # ── SQL ─────────────────────────────────────────────────
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
        # ── Bash ────────────────────────────────────────────────
        ("if [ -f file ]; then echo found; fi", {"if", "then", "echo", "fi"}),
        ("for f in *.py; do echo $f; done", {"for", "f", "in", "do", "echo", "done"}),
        ("while read line; do process; done", {"while", "read", "line", "do", "done"}),
        ("case $1 in start) ;; esac", {"case", "in", "start", "esac"}),
    ],
    ids=lambda val: str(sorted(val) if isinstance(val, set) else val)[:50],
)
def test_keywords_survive_tokenisation(code: str, expected_keywords: set[str]) -> None:
    """Keywords appear in tokenise_code output."""
    tokens = tokenise_code(code)
    for kw in expected_keywords:
        assert kw in tokens.split(), f"{kw!r} missing from {tokens!r}"


@fixture
@parametrize_with_cases(
    "case",
    cases=".cases_tokenise_keywords",
    prefix="case_",
)
def keyword_store_and_case(case: KeywordScenario) -> Generator[tuple[IndexStore, KeywordScenario]]:
    """Seed a fresh store with the single chunk the case describes."""
    chunk = make_chunk(
        case.chunk_id,
        name=case.name,
        content=case.content,
        path=f"src/{case.chunk_id}.py",
    )
    store = IndexStore(writable=True)
    with store.session() as ws:
        ws.add_chunk(chunk)
        ws.insert_snapshots(
            [Snapshot(commit_sha="head", file_path=chunk.file_path, blob_sha=chunk.blob_sha)],
            repo_id=1,
        )
    yield store, case
    store.close()


def test_keyword_searchable_via_fts(
    keyword_store_and_case: tuple[IndexStore, KeywordScenario],
) -> None:
    """Every keyword the case declares is findable via FTS."""
    store, case = keyword_store_and_case
    results = store.match_fulltext("head", case.keyword, top_k=5, repo_id=1)
    assert len(results) > 0, f"FTS miss for keyword {case.keyword!r} in chunk {case.chunk_id!r}"
    assert results[0][0].id == case.chunk_id

"""Tests for code-aware tokenisation."""

from __future__ import annotations

import pytest

from rbtr.index.tokenise import tokenise_code


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

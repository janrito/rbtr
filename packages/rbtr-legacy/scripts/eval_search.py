#!/usr/bin/env python3
"""Evaluate search quality against curated queries.

The eval queries are written against the **rbtr** repo itself
(always available, well-understood).  Running against a different
repo will skip all queries and exit with a warning.

Usage::

    just eval-search                   # current repo at HEAD (no embeddings)
    just eval-search -- --embed        # with embeddings (slower, runs T5 queries)
    just eval-search -- . main         # current repo at a specific ref
    just eval-search -- --embed . main # specific ref with embeddings
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from rbtr_legacy.index.store import IndexStore

# ── Eval data model ──────────────────────────────────────────────────


class Technique(StrEnum):
    """Which search improvement technique a query tests."""

    T1_TOKENISATION = "T1"
    T2_IDF = "T2"
    T3_KIND_SCORING = "T3"
    T4_FILE_CATEGORY = "T4"
    T5_HYBRID_FUSION = "T5"
    T6_NAME_MATCH = "T6"
    T7_QUERY_UNDERSTANDING = "T7"
    T8_STRUCTURAL = "T8"
    T9_DIFF_PROXIMITY = "T9"


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query with expected results."""

    id: str
    query: str
    expected: list[str]
    """Expected results as `file_path_suffix:symbol_name` pairs.

    A result matches if its file_path ends with the path part
    and its name matches the name part.  This avoids hardcoding
    full paths like `src/rbtr/engine/agent.py`.
    """
    technique: Technique
    needs_embeddings: bool = False
    needs_diff: bool = False
    changed_files: set[str] | None = None
    """File paths for the simulated diff (T9 queries only)."""


# ── Eval queries ─────────────────────────────────────────────────────
# Drawn from TODO-search.md.  Each query targets a specific weakness.

EVAL_QUERIES: list[EvalQuery] = [
    # T1 — Code-aware tokenisation
    EvalQuery(
        id="1",
        query="AgentDeps",
        expected=["engine/agent.py:AgentDeps"],
        technique=Technique.T1_TOKENISATION,
    ),
    EvalQuery(
        id="2",
        query="ModelMessage",
        expected=[
            # ModelMessage is an external type (pydantic_ai) used inside
            # function bodies.  The best search can do is find import
            # chunks that reference it.  Finding functions that USE it
            # is find_references territory, not search.
            "engine/compact.py:from pydantic_ai.messages import ModelMessage",
            "engine/history.py:from pydantic_ai.messages import (",
        ],
        technique=Technique.T1_TOKENISATION,
    ),
    EvalQuery(
        id="3",
        query="build_model",
        expected=[
            "providers/__init__.py:build_model",
            "providers/claude.py:build_model",
            "providers/openai.py:build_model",
        ],
        technique=Technique.T1_TOKENISATION,
    ),
    EvalQuery(
        id="4",
        query="RunContext",
        expected=[
            # RunContext is an external type (pydantic_ai).  The best
            # search result is a source-file import that references it.
            "engine/tools.py:from pydantic_ai import RunContext",
            "engine/agent.py:from pydantic_ai import Agent, RunContext",
        ],
        technique=Technique.T1_TOKENISATION,
    ),
    EvalQuery(
        id="5",
        query="ThinkingEffort",
        expected=["config.py:ThinkingEffort"],
        technique=Technique.T1_TOKENISATION,
    ),
    EvalQuery(
        id="6",
        query="_deep_merge",
        expected=["config.py:_deep_merge"],
        technique=Technique.T1_TOKENISATION,
    ),
    # T2 — IDF neutralisation
    EvalQuery(
        id="7",
        query="config",
        expected=["config.py:Config", "config.py:IndexConfig"],
        technique=Technique.T2_IDF,
    ),
    EvalQuery(
        id="8",
        query="import edge",
        expected=["index/edges.py:infer_import_edges"],
        technique=Technique.T2_IDF,
    ),
    EvalQuery(
        id="9",
        query="model",
        expected=[
            # "model" is a prefix match for ModelMetadata (CLASS) — the
            # most specific match.  build_model is a substring match that
            # legitimately ranks lower.
            "providers/endpoint.py:ModelMetadata",
            "providers/openai_codex.py:ModelMetadata",
        ],
        technique=Technique.T2_IDF,
    ),
    EvalQuery(
        id="9a",
        query="store chunk",
        expected=[
            "index/store.py:insert_chunks",
            "index/store.py:get_chunks",
        ],
        technique=Technique.T2_IDF,
    ),
    EvalQuery(
        id="9b",
        query="class config",
        expected=["config.py:Config"],
        technique=Technique.T2_IDF,
    ),
    # T3 — Symbol-kind scoring
    EvalQuery(
        id="10",
        query="Engine",
        expected=["engine/core.py:Engine"],
        technique=Technique.T3_KIND_SCORING,
    ),
    EvalQuery(
        id="11",
        query="SessionStore",
        expected=["sessions/store.py:SessionStore"],
        technique=Technique.T3_KIND_SCORING,
    ),
    EvalQuery(
        id="12",
        query="chunk_markdown",
        expected=["index/chunks.py:chunk_markdown"],
        technique=Technique.T3_KIND_SCORING,
    ),
    # T4 — File category penalty
    EvalQuery(
        id="13",
        query="infer_import_edges",
        expected=["index/edges.py:infer_import_edges"],
        technique=Technique.T4_FILE_CATEGORY,
    ),
    EvalQuery(
        id="14",
        query="build_index",
        expected=["index/orchestrator.py:build_index"],
        technique=Technique.T4_FILE_CATEGORY,
    ),
    EvalQuery(
        id="15",
        query="search_fulltext",
        expected=["index/store.py:search_fulltext"],
        technique=Technique.T4_FILE_CATEGORY,
    ),
    # T5 — Hybrid fusion / semantic + lexical
    #
    # Each target has 3 paraphrases (short / medium / long) to
    # distinguish "hard target" (all phrasings fail) from "hard
    # phrasing" (some work, some don't).  IDs use the format
    # <group>.<variant> — e.g. 16a/16b/16c target the same code.
    #
    # ── handle_llm / render_system ───────────────────────────
    EvalQuery(
        id="16a",
        query="send messages to the LLM",
        expected=["engine/llm.py:handle_llm"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="16b",
        query="how does the review context get sent to the llm",
        expected=[
            "engine/llm.py:handle_llm",
            "prompts/__init__.py:render_system",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="16c",
        query="stream a user message through the active model and collect the response",
        expected=["engine/llm.py:handle_llm"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── SessionStore ─────────────────────────────────────────
    EvalQuery(
        id="17a",
        query="session persistence",
        expected=["sessions/store.py:SessionStore"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="17b",
        query="database storage for sessions",
        expected=["sessions/store.py:SessionStore"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="17c",
        query="where is the conversation history stored between runs",
        expected=["sessions/store.py:SessionStore"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── save_draft ───────────────────────────────────────────
    EvalQuery(
        id="18a",
        query="persist draft to disk",
        expected=["github/draft.py:save_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="18b",
        query="where are draft comments saved",
        expected=["github/draft.py:save_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="18c",
        query="write the current review draft as yaml to the local filesystem",
        expected=["github/draft.py:save_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── compact_history / _auto_compact_on_overflow ──────────
    EvalQuery(
        id="31a",
        query="summarise old messages",
        expected=[
            "engine/compact.py:compact_history",
            "engine/compact.py:compact_history_async",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="31b",
        query="shorten the conversation when it gets too long",
        expected=[
            "engine/compact.py:compact_history",
            "engine/llm.py:_auto_compact_on_overflow",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="31c",
        query="reduce context usage by compressing earlier turns into a summary",
        expected=[
            "engine/compact.py:compact_history",
            "engine/compact.py:compact_history_async",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── estimate_tokens ──────────────────────────────────────
    EvalQuery(
        id="32a",
        query="token count",
        expected=["engine/history.py:estimate_tokens"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="32b",
        query="count how many tokens a message uses",
        expected=["engine/history.py:estimate_tokens"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="32c",
        query="approximate the number of tokens in a list of chat messages",
        expected=["engine/history.py:estimate_tokens"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── changed_files ────────────────────────────────────────
    EvalQuery(
        id="35a",
        query="modified files in the diff",
        expected=["engine/tools.py:changed_files"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="35b",
        query="figure out which files were modified in the PR",
        expected=[
            "engine/tools.py:changed_files",
            "engine/review.py:_get_diff_ranges",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="35c",
        query="list every file path that has changes between the base and head commits",
        expected=[
            "engine/tools.py:changed_files",
            "git/objects.py:changed_files",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── post_review_draft / sync_review_draft ────────────────
    EvalQuery(
        id="36a",
        query="post review to github",
        expected=[
            "engine/review.py:post_review_draft",
            "github/client.py:post_review",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="36b",
        query="submit review comments to github",
        expected=[
            "engine/review.py:post_review_draft",
            "engine/review.py:sync_review_draft",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="36c",
        query="push the locally drafted review comments to the remote pull request",
        expected=[
            "engine/review.py:post_review_draft",
            "engine/review.py:sync_review_draft",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── system_prompt / render_system ────────────────────────
    EvalQuery(
        id="37a",
        query="build the system prompt",
        expected=[
            "engine/agent.py:system_prompt",
            "prompts/__init__.py:render_system",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="37b",
        query="render the system message for the LLM",
        expected=[
            "engine/agent.py:system_prompt",
            "prompts/__init__.py:render_system",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="37c",
        query="assemble the instructions that go at the start of every conversation",
        expected=[
            "engine/agent.py:system_prompt",
            "prompts/__init__.py:render_system",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── _repair_dangling_tool_calls ──────────────────────────
    EvalQuery(
        id="43a",
        query="fix orphaned tool calls",
        expected=["engine/llm.py:_repair_dangling_tool_calls"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="43b",
        query="fix broken tool call history after cancellation",
        expected=["engine/llm.py:_repair_dangling_tool_calls"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="43c",
        query="clean up the message list when a tool-calling turn was interrupted",
        expected=["engine/llm.py:_repair_dangling_tool_calls"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── serialise_for_summary ────────────────────────────────
    EvalQuery(
        id="44a",
        query="format messages as text",
        expected=["engine/history.py:serialise_for_summary"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="44b",
        query="convert messages to readable text for summarisation",
        expected=["engine/history.py:serialise_for_summary"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="44c",
        query="turn the structured chat history into a plain-text transcript for the compaction prompt",
        expected=["engine/history.py:serialise_for_summary"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── _deep_merge ──────────────────────────────────────────
    EvalQuery(
        id="45a",
        query="merge nested dicts",
        expected=["config.py:_deep_merge"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="45b",
        query="recursively merge two config dictionaries",
        expected=["config.py:_deep_merge"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="45c",
        query="combine user settings with workspace overrides by walking nested keys",
        expected=["config.py:_deep_merge"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── split_history ────────────────────────────────────────
    EvalQuery(
        id="48a",
        query="partition conversation messages",
        expected=["engine/history.py:split_history"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="48b",
        query="divide history into old and recent messages",
        expected=["engine/history.py:split_history"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="48c",
        query="separate the message list into a prefix to be summarised and a suffix to keep",
        expected=["engine/history.py:split_history"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── demote_thinking ──────────────────────────────────────
    EvalQuery(
        id="49a",
        query="strip thinking from responses",
        expected=["engine/history.py:demote_thinking"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="49b",
        query="remove thinking blocks from model output",
        expected=["engine/history.py:demote_thinking"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="49c",
        query="convert ThinkingParts in the history into plain TextParts so older turns are cheaper",
        expected=["engine/history.py:demote_thinking"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── run_index / _build_index ─────────────────────────────
    EvalQuery(
        id="47a",
        query="kick off indexing",
        expected=[
            "engine/indexing.py:run_index",
            "engine/indexing.py:_build_index",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="47b",
        query="start background indexing for a repository",
        expected=[
            "engine/indexing.py:run_index",
            "engine/indexing.py:_build_index",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="47c",
        query="launch the daemon thread that builds the code index for the current review target",
        expected=[
            "engine/indexing.py:run_index",
            "engine/indexing.py:_build_index",
        ],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── tokenise_code ────────────────────────────────────────
    EvalQuery(
        id="40a",
        query="break apart identifiers",
        expected=["index/tokenise.py:tokenise_code"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="40b",
        query="split camelCase identifiers for search",
        expected=["index/tokenise.py:tokenise_code"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="40c",
        query="expand compound names like getUserById into individual words for the FTS index",
        expected=["index/tokenise.py:tokenise_code"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # ── load_draft ───────────────────────────────────────────
    EvalQuery(
        id="50a",
        query="read draft from disk",
        expected=["github/draft.py:load_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="50b",
        query="load a saved review from the filesystem",
        expected=["github/draft.py:load_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="50c",
        query="deserialise the yaml review draft file and return the ReviewDraft object",
        expected=["github/draft.py:load_draft"],
        technique=Technique.T5_HYBRID_FUSION,
        needs_embeddings=True,
    ),
    # T6 — Name-match boosting
    EvalQuery(
        id="19",
        query="_embed_missing",
        expected=["index/orchestrator.py:_embed_missing"],
        technique=Technique.T6_NAME_MATCH,
    ),
    EvalQuery(
        id="20",
        query="IndexStore",
        expected=["index/store.py:IndexStore"],
        technique=Technique.T6_NAME_MATCH,
    ),
    EvalQuery(
        id="21",
        query="cmd_draft",
        expected=["engine/draft_cmd.py:cmd_draft"],
        technique=Technique.T6_NAME_MATCH,
    ),
    # T7 — Query understanding
    EvalQuery(
        id="22",
        query="what happens when the context window overflows",
        expected=[
            "engine/llm.py:_auto_compact_on_overflow",
            "engine/compact.py:compact_history",
        ],
        technique=Technique.T7_QUERY_UNDERSTANDING,
        needs_embeddings=True,
    ),
    EvalQuery(
        id="23",
        query="estimate_tokens",
        expected=["engine/history.py:estimate_tokens"],
        technique=Technique.T7_QUERY_UNDERSTANDING,
    ),
    # T8 — Structural signals
    EvalQuery(
        id="25",
        query="Chunk",
        expected=["index/models.py:Chunk"],
        technique=Technique.T8_STRUCTURAL,
    ),
    EvalQuery(
        id="26",
        query="store",
        expected=["index/store.py:IndexStore"],
        technique=Technique.T8_STRUCTURAL,
    ),
    EvalQuery(
        id="27",
        query="embedding",
        expected=[
            "index/embeddings.py:embed_texts",
            "index/store.py:update_embeddings",
        ],
        technique=Technique.T8_STRUCTURAL,
    ),
]

# T9 — proximity-to-diff queries.  Simulate a diff touching
# index/store.py.
_T9_CHANGED_FILES = {
    "src/rbtr/index/store.py",
}

EVAL_QUERIES += [
    EvalQuery(
        id="28",
        query="search",
        expected=[
            "index/store.py:search",
            "index/store.py:search_fulltext",
            "index/store.py:search_by_name",
            "index/store.py:search_similar",
        ],
        technique=Technique.T9_DIFF_PROXIMITY,
        needs_diff=True,
        changed_files=_T9_CHANGED_FILES,
    ),
    EvalQuery(
        id="29",
        query="insert",
        expected=[
            "index/store.py:insert_chunks",
            "index/store.py:insert_edges",
        ],
        technique=Technique.T9_DIFF_PROXIMITY,
        needs_diff=True,
        changed_files=_T9_CHANGED_FILES,
    ),
    EvalQuery(
        id="30",
        query="checkpoint",
        expected=["index/store.py:checkpoint"],
        technique=Technique.T9_DIFF_PROXIMITY,
        needs_diff=True,
        changed_files=_T9_CHANGED_FILES,
    ),
]

# #24 (pattern query) requires runtime context that a static
# eval harness can't provide.


# ── Metrics ──────────────────────────────────────────────────────────


def _matches(chunk_file_path: str, chunk_name: str, expected: str) -> bool:
    """Check if a chunk matches an expected `path_suffix:name` spec."""
    path_part, name_part = expected.rsplit(":", 1)
    return chunk_file_path.endswith(path_part) and chunk_name == name_part


def _find_rank(
    results: list[tuple[str, str]],
    expected: list[str],
) -> int | None:
    """Find the rank (1-indexed) of the first matching result.

    Returns None if no expected result appears in the list.
    """
    for rank, (file_path, name) in enumerate(results, 1):
        for exp in expected:
            if _matches(file_path, name, exp):
                return rank
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    """1.0 if rank <= k, else 0.0.  None rank = 0.0."""
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """1/rank, or 0.0 if not found."""
    if rank is None:
        return 0.0
    return 1.0 / rank


# ── Result tracking ─────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Result of running one eval query against one backend."""

    query_id: str
    technique: Technique
    backend: str
    rank: int | None
    top_results: list[str]


@dataclass
class EvalRun:
    """Aggregate results from one evaluation run."""

    results: list[QueryResult] = field(default_factory=list)

    def add(self, r: QueryResult) -> None:
        self.results.append(r)

    def metrics_by_technique(self, backend: str) -> dict[Technique, dict[str, float]]:
        """Compute per-technique recall@1, recall@5, MRR."""
        from collections import defaultdict

        groups: dict[Technique, list[QueryResult]] = defaultdict(list)
        for r in self.results:
            if r.backend == backend:
                groups[r.technique].append(r)

        metrics: dict[Technique, dict[str, float]] = {}
        for tech, qrs in sorted(groups.items(), key=lambda x: x[0].value):
            n = len(qrs)
            r1 = sum(recall_at_k(q.rank, 1) for q in qrs) / n
            r5 = sum(recall_at_k(q.rank, 5) for q in qrs) / n
            mrr = sum(reciprocal_rank(q.rank) for q in qrs) / n
            metrics[tech] = {"R@1": r1, "R@5": r5, "MRR": mrr, "n": n}
        return metrics

    def overall_metrics(self, backend: str) -> dict[str, float]:
        """Compute overall recall@1, recall@5, MRR for a backend."""
        relevant = [r for r in self.results if r.backend == backend]
        if not relevant:
            return {"R@1": 0.0, "R@5": 0.0, "MRR": 0.0, "n": 0}
        n = len(relevant)
        return {
            "R@1": sum(recall_at_k(r.rank, 1) for r in relevant) / n,
            "R@5": sum(recall_at_k(r.rank, 5) for r in relevant) / n,
            "MRR": sum(reciprocal_rank(r.rank) for r in relevant) / n,
            "n": n,
        }


# ── Search runners ───────────────────────────────────────────────────


def _run_name_search(
    store: IndexStore,
    commit_sha: str,
    query: str,
    top_k: int = 10,
) -> list[tuple[str, str]]:
    """Run search_by_name and return (file_path, name) pairs."""
    chunks = store.search_by_name(commit_sha, query)
    return [(c.file_path, c.name) for c in chunks[:top_k]]


def _run_fulltext_search(
    store: IndexStore,
    commit_sha: str,
    query: str,
    top_k: int = 10,
) -> list[tuple[str, str]]:
    """Run search_fulltext (BM25) and return (file_path, name) pairs."""
    try:
        results = store.search_fulltext(commit_sha, query, top_k=top_k)
        return [(c.file_path, c.name) for c, _score in results]
    except Exception as exc:
        print(f"    BM25 error: {exc}")
        return []


def _run_semantic_search(
    store: IndexStore,
    commit_sha: str,
    query: str,
    top_k: int = 10,
) -> list[tuple[str, str]]:
    """Run search_by_text (embedding cosine) and return (file_path, name) pairs."""
    try:
        results = store.search_by_text(commit_sha, query, top_k=top_k)
        return [(c.file_path, c.name) for c, _score in results]
    except Exception:
        return []


def _run_unified_search(
    store: IndexStore,
    commit_sha: str,
    query: str,
    top_k: int = 10,
    changed_files: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Run unified search() and return (file_path, name) pairs."""
    try:
        results = store.search(commit_sha, query, top_k=top_k, changed_files=changed_files)
        return [(r.chunk.file_path, r.chunk.name) for r in results]
    except Exception as exc:
        print(f"    unified error: {exc}")
        return []


# ── Reporting ────────────────────────────────────────────────────────


def _print_header(title: str) -> None:
    print(f"\n{'═' * 4} {title} {'═' * (60 - len(title))}")


def _print_table(
    run: EvalRun,
    backend: str,
) -> None:
    """Print per-technique metrics table for one backend."""
    metrics = run.metrics_by_technique(backend)
    overall = run.overall_metrics(backend)

    print(f"  {'Tech':<6} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'n':>4}")
    print(f"  {'─' * 28}")
    for tech, m in metrics.items():
        r1 = f"{m['R@1']:.0%}"
        r5 = f"{m['R@5']:.0%}"
        mrr = f"{m['MRR']:.2f}"
        n = f"{m['n']:.0f}"
        print(f"  {tech.value:<6} {r1:>6} {r5:>6} {mrr:>6} {n:>4}")
    print(f"  {'─' * 28}")
    r1 = f"{overall['R@1']:.0%}"
    r5 = f"{overall['R@5']:.0%}"
    mrr = f"{overall['MRR']:.2f}"
    n = f"{overall['n']:.0f}"
    print(f"  {'ALL':<6} {r1:>6} {r5:>6} {mrr:>6} {n:>4}")


def _print_misses(run: EvalRun, backend: str) -> None:
    """Print queries where the backend failed to rank the target in top 5."""
    misses = [r for r in run.results if r.backend == backend and (r.rank is None or r.rank > 5)]
    if not misses:
        return
    print("\n  Misses (not in top 5):")
    for m in misses:
        query_obj = next(q for q in EVAL_QUERIES if q.id == m.query_id)
        actual = ", ".join(m.top_results[:3]) if m.top_results else "(no results)"
        rank_str = f"rank={m.rank}" if m.rank else "not found"
        print(f"    [{m.technique.value}] q={query_obj.query!r}  {rank_str}")
        print(f"         expected: {query_obj.expected}")
        print(f"         got:      {actual}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import pygit2

    from rbtr_legacy.index.orchestrator import build_index
    from rbtr_legacy.index.store import IndexStore

    # Parse args.  Strip leading `--` that `just` passes through.
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]

    embed = False
    positional: list[str] = []
    for a in args:
        if a == "--embed":
            embed = True
        else:
            positional.append(a)
    repo_path = Path(positional[0]) if positional else Path(".")
    ref = positional[1] if len(positional) > 1 else "HEAD"

    repo = pygit2.Repository(str(repo_path))
    commit = repo.revparse_single(ref)
    commit_sha = str(commit.id)

    # Verify this is the rbtr repo — the eval queries are rbtr-specific.
    pyproject = repo_path.resolve() / "pyproject.toml"
    if not pyproject.exists() or 'name = "rbtr"' not in pyproject.read_text():
        print(
            "Error: eval queries are written against the rbtr repo.\n"
            f"  {repo_path.resolve()} does not appear to be rbtr.\n"
            "  Run without arguments to use the current repo, or\n"
            "  point at a checkout of rbtr."
        )
        sys.exit(1)

    print(f"Repository: {repo_path.resolve()}")
    print(f"Ref:        {ref} ({commit_sha[:12]})")
    print(f"Embeddings: {'yes' if embed else 'no (use --embed to enable)'}")

    # Build index.
    _print_header("Indexing")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "eval.duckdb"
        store = IndexStore(db_path)

        t0 = time.monotonic()
        if embed:
            result = build_index(repo, commit_sha, store)
        else:
            with patch("rbtr.index.orchestrator._embed_missing"):
                result = build_index(repo, commit_sha, store)
        elapsed = time.monotonic() - t0

        embed_label = "" if embed else " (no embeddings)"
        print(
            f"  {result.stats.total_chunks} chunks, "
            f"{result.stats.total_edges} edges, "
            f"{elapsed:.1f}s{embed_label}"
        )

        # Run eval queries.
        _print_header("Evaluation")

        type SearchFn = Callable[
            [IndexStore, str, str, int],
            list[tuple[str, str]],
        ]

        run = EvalRun()
        backends: dict[str, SearchFn] = {
            "name": _run_name_search,
            "bm25": _run_fulltext_search,
            "unified": _run_unified_search,  # type: ignore[dict-item]  # unified has extra kwarg
        }
        if embed:
            backends["semantic"] = _run_semantic_search

        # When embeddings are available, run all queries.
        # Otherwise skip embedding-dependent ones.
        if embed:
            runnable = list(EVAL_QUERIES)
            skipped = 0
        else:
            runnable = [q for q in EVAL_QUERIES if not q.needs_embeddings]
            skipped = len(EVAL_QUERIES) - len(runnable)
        diff_queries = [q for q in runnable if q.needs_diff]
        non_diff = [q for q in runnable if not q.needs_diff]
        parts = [f"{len(runnable)} queries"]
        if diff_queries:
            parts.append(f"{len(diff_queries)} T9 diff-context")
        if skipped:
            parts.append(f"skipping {skipped} that need embeddings")
        print(f"  Running {', '.join(parts)}")

        for eq in non_diff:
            for backend_name, search_fn in backends.items():
                results = search_fn(store, commit_sha, eq.query, top_k=10)
                rank = _find_rank(results, eq.expected)
                top_strs = [f"{fp}:{n}" for fp, n in results[:5]]
                run.add(
                    QueryResult(
                        query_id=eq.id,
                        technique=eq.technique,
                        backend=backend_name,
                        rank=rank,
                        top_results=top_strs,
                    )
                )

        # T9 diff-context queries — unified backend only.
        for eq in diff_queries:
            results = _run_unified_search(
                store,
                commit_sha,
                eq.query,
                top_k=10,
                changed_files=eq.changed_files,
            )
            rank = _find_rank(results, eq.expected)
            top_strs = [f"{fp}:{n}" for fp, n in results[:5]]
            run.add(
                QueryResult(
                    query_id=eq.id,
                    technique=eq.technique,
                    backend="unified",
                    rank=rank,
                    top_results=top_strs,
                )
            )

        # Report.
        for backend_name in backends:
            _print_header(f"Backend: {backend_name}")
            _print_table(run, backend_name)
            _print_misses(run, backend_name)

        store.close()

    print()


if __name__ == "__main__":
    main()

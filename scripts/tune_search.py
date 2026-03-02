#!/usr/bin/env python3
"""Tune unified search fusion weights via grid search.

Builds the index once, precomputes all channel scores (BM25, name,
importance, proximity) for every eval query, then sweeps weight
combinations in-memory.  Reports the best weights per query kind
(identifier / concept).

Usage::

    just tune-search              # current repo at HEAD
    just tune-search -- --step 0.05   # finer resolution

The sweep is fast because channel scores are precomputed once —
only the fusion arithmetic is repeated per combo.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Re-use eval infrastructure.
from eval_search import (
    EVAL_QUERIES,
    EvalQuery,
    _find_rank,
    _print_header,
)


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    """Generate all (alpha, beta, gamma) triples that sum to 1.0."""
    n = round(1.0 / step)
    combos: list[tuple[float, float, float]] = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            g = n - a - b
            combos.append((a * step, b * step, g * step))
    return combos


def _reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def _recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def main() -> None:
    import pygit2

    from rbtr.index.models import Chunk
    from rbtr.index.orchestrator import build_index
    from rbtr.index.search import (
        QueryKind,
        classify_query,
        fuse_scores,
        importance_score,
        name_score,
        proximity_score,
    )
    from rbtr.index.store import IndexStore

    # ── Parse args ───────────────────────────────────────────
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]

    step = 0.1
    embed = False
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--step" and i + 1 < len(args):
            step = float(args[i + 1])
            i += 2
        elif args[i] == "--embed":
            embed = True
            i += 1
        else:
            positional.append(args[i])
            i += 1

    repo_path = Path(positional[0]) if positional else Path(".")
    ref = positional[1] if len(positional) > 1 else "HEAD"

    repo = pygit2.Repository(str(repo_path))
    commit = repo.revparse_single(ref)
    commit_sha = str(commit.id)

    pyproject = repo_path.resolve() / "pyproject.toml"
    if not pyproject.exists() or 'name = "rbtr"' not in pyproject.read_text():
        print("Error: must run against the rbtr repo.")
        sys.exit(1)

    print(f"Repository: {repo_path.resolve()}")
    print(f"Ref:        {ref} ({commit_sha[:12]})")
    print(f"Step:       {step}")
    print(f"Embeddings: {'yes' if embed else 'no'}")

    # ── Build index once ─────────────────────────────────────
    _print_header("Indexing")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "tune.duckdb"
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

        # ── Precompute channel scores for each query ─────────
        _print_header("Precomputing channel scores")

        # Per-query precomputed data.
        type PrecomputedQuery = tuple[
            EvalQuery,
            QueryKind,
            dict[str, Chunk],  # candidates
            dict[str, float],  # lexical_scores
            dict[str, float],  # semantic_scores
            dict[str, float],  # name_scores
            dict[str, float],  # importance_scores
            dict[str, float] | None,  # proximity_scores
        ]

        precomputed: list[PrecomputedQuery] = []
        pool_size = 30

        if embed:
            runnable = list(EVAL_QUERIES)
        else:
            runnable = [q for q in EVAL_QUERIES if not q.needs_embeddings]

        # Load all edges once for proximity calculations.
        all_edges = store.get_edges(commit_sha)

        for eq in runnable:
            kind = classify_query(eq.query)
            candidates: dict[str, Chunk] = {}
            lexical_scores: dict[str, float] = {}

            # BM25 channel.
            for chunk, score in store.search_fulltext(commit_sha, eq.query, top_k=pool_size):
                lexical_scores[chunk.id] = score
                candidates[chunk.id] = chunk

            # Semantic channel (embedding cosine) — only if embeddings computed.
            semantic_scores: dict[str, float] = {}
            if embed:
                try:
                    for chunk, score in store.search_by_text(commit_sha, eq.query, top_k=pool_size):
                        semantic_scores[chunk.id] = score
                        if chunk.id not in candidates:
                            candidates[chunk.id] = chunk
                except Exception:  # noqa: S110  # best-effort: skip if model unavailable
                    pass

            # Name channel — full query + per-token expansion.
            for chunk in store.search_by_name(commit_sha, eq.query):
                if chunk.id not in candidates:
                    candidates[chunk.id] = chunk

            tokens = eq.query.split()
            if len(tokens) > 1:
                for token in tokens:
                    if len(token) >= 3:
                        for chunk in store.search_by_name(commit_sha, token):
                            if chunk.id not in candidates:
                                candidates[chunk.id] = chunk

            # Compute name scores for all candidates.
            name_scores: dict[str, float] = {}
            for cid, chunk in candidates.items():
                ns = name_score(eq.query, chunk.name)
                if ns > 0.0:
                    name_scores[cid] = ns

            # Importance (inbound-degree).
            candidate_ids = list(candidates.keys())
            degrees = store.inbound_degrees(commit_sha, candidate_ids)
            imp_scores: dict[str, float] = {
                cid: importance_score(degrees.get(cid, 0)) for cid in candidate_ids
            }

            # Proximity (diff distance) — only for queries with changed_files.
            prox_scores: dict[str, float] | None = None
            if eq.changed_files:
                candidate_set = set(candidate_ids)
                neighbours: dict[str, set[str]] = {cid: set() for cid in candidate_set}
                for e in all_edges:
                    if e.source_id in candidate_set:
                        neighbours[e.source_id].add(e.target_id)
                    if e.target_id in candidate_set:
                        neighbours[e.target_id].add(e.source_id)

                # Resolve neighbour file paths from candidates.
                nb_paths: dict[str, str] = {cid: c.file_path for cid, c in candidates.items()}

                has_edge: set[str] = set()
                for cid, nbs in neighbours.items():
                    for nb in nbs:
                        if nb_paths.get(nb, "") in eq.changed_files:
                            has_edge.add(cid)
                            break

                prox_scores = {
                    cid: proximity_score(
                        chunk.file_path,
                        eq.changed_files,
                        has_edge_to_changed=cid in has_edge,
                    )
                    for cid, chunk in candidates.items()
                }

            precomputed.append(
                (
                    eq,
                    kind,
                    candidates,
                    lexical_scores,
                    semantic_scores,
                    name_scores,
                    imp_scores,
                    prox_scores,
                )
            )

        print(f"  {len(precomputed)} queries precomputed")

        # ── Grid search per query kind ───────────────────────
        grid = _weight_grid(step)
        print(f"  {len(grid)} weight combos per kind (step={step})")

        for target_kind in (QueryKind.IDENTIFIER, QueryKind.CONCEPT):
            queries = [
                (eq, cands, lex, sem, names, imp, prox)
                for eq, kind, cands, lex, sem, names, imp, prox in precomputed
                if kind == target_kind
            ]
            if not queries:
                continue

            _print_header(f"Tuning {target_kind.value} ({len(queries)} queries)")

            type WeightResult = tuple[float, float, float, float, float, float]
            all_results: list[WeightResult] = []

            t0 = time.monotonic()
            for alpha, beta, gamma in grid:
                mrr_sum = 0.0
                r1_sum = 0.0
                r5_sum = 0.0

                for eq, cands, lex_scores, sem_scores, nm_scores, imp_s, prox_s in queries:
                    scored = fuse_scores(
                        cands,
                        lex_scores,
                        sem_scores,
                        nm_scores,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        top_k=10,
                        importance_scores=imp_s,
                        proximity_scores=prox_s,
                    )
                    results = [(r.chunk.file_path, r.chunk.name) for r in scored]
                    rank = _find_rank(results, eq.expected)
                    mrr_sum += _reciprocal_rank(rank)
                    r1_sum += _recall_at_k(rank, 1)
                    r5_sum += _recall_at_k(rank, 5)

                n = len(queries)
                all_results.append(
                    (
                        alpha,
                        beta,
                        gamma,
                        mrr_sum / n,
                        r1_sum / n,
                        r5_sum / n,
                    )
                )

            elapsed = time.monotonic() - t0

            # Sort by MRR desc, then R@1 desc.
            all_results.sort(key=lambda x: (x[3], x[4]), reverse=True)

            print(f"\n  Top 10 ({elapsed:.1f}s):")
            print(f"  {'a(sem)':>7} {'b(lex)':>7} {'g(name)':>7} {'MRR':>6} {'R@1':>6} {'R@5':>6}")
            print(f"  {'─' * 43}")
            for a, b, g, mrr, r1, r5 in all_results[:10]:
                print(f"  {a:7.2f} {b:7.2f} {g:7.2f} {mrr:6.3f} {r1:5.0%} {r5:5.0%}")

            a, b, g, mrr, r1, r5 = all_results[0]
            print(f"\n  Best: a={a:.2f}  b={b:.2f}  g={g:.2f}  (MRR={mrr:.3f}, R@1={r1:.0%})")

        # ── Current weights for reference ────────────────────
        _print_header("Current weights")
        from rbtr.index.search import _KIND_WEIGHTS

        for kind, weights in _KIND_WEIGHTS.items():
            a, b, g = weights
            print(f"  {kind.value:<12} a={a:.2f}  b={b:.2f}  g={g:.2f}")

        store.close()

    print()


if __name__ == "__main__":
    main()

# Index profiling results

Benchmark results for the rbtr code index pipeline. All
measurements taken on Apple M3 Pro (18 GB), macOS, Python
3.13, DuckDB 1.2, with `llama-cpp-python` (Metal).

## Test repos

| Repo     | Files | Chunks | Top languages                 |
| -------- | ----: | -----: | ----------------------------- |
| rbtr     |   131 |  2,264 | Python 107, Markdown 6        |
| typeshed | 3,319 | 64,074 | Python 3070, TOML 125         |
| ukf      | 4,462 | 39,287 | Python 1574, HTML 579, JS 435 |

## Indexing time (no embedding)

End-to-end `build_index()` with embedding stubbed out.

| Repo     | Total |     Listing |  Extraction |   DB insert |      Edges |
| -------- | ----: | ----------: | ----------: | ----------: | ---------: |
| rbtr     | 0.28s |  49ms (18%) | 170ms (61%) |  32ms (11%) |   7ms (3%) |
| typeshed | 1.94s |  150ms (8%) | 925ms (48%) | 305ms (16%) | 105ms (5%) |
| ukf      | 3.21s | 1.09s (34%) | 1.35s (42%) | 349ms (11%) | 188ms (6%) |

FTS index is rebuilt lazily on first keyword search, not during
indexing.

### Optimizations applied

Three optimizations produced a **51% speed-up** on the ukf
structural index (6.52s → 3.21s) and eliminated a **43-minute
DB write bottleneck** in the embedding pipeline:

1. **Cached tree-sitter `Query()` per language.** `Query()`
   compilation costs ~0.6 ms each. Recompiling the same
   query for every file of the same language wasted ~0.9s
   on 1,574 Python files. `lru_cache(32)` in
   `treesitter._get_query()` eliminates the redundancy.

2. **Deferred FTS rebuild.** `store.rebuild_fts_index()` takes
   ~2.4s on 39K chunks. The store already tracks `_fts_dirty`
   and lazily rebuilds on first `search_fulltext()` call.
   Removing the explicit call from `build_index` / `update_index`
   moves this cost from indexing time to first-search time.

3. **Batch embedding writes via PyArrow join-UPDATE.**
   Individual `UPDATE chunks SET embedding = ? WHERE id = ?`
   costs ~67 ms/call due to DuckDB per-statement overhead.
   For 39K chunks this alone took **43 minutes**. Replaced
   with `update_embeddings()` which registers a PyArrow table
   and runs a single join-UPDATE — **0.27 ms/call** (248x
   faster). Projected DB write time: **10 seconds**.

### Per-file and per-chunk rates

| Repo     | Per file | Per chunk |
| -------- | -------: | --------: |
| rbtr     |    1.3ms |    0.07ms |
| typeshed |    0.6ms |    0.03ms |
| ukf      |    0.3ms |    0.03ms |

## Query latency

All queries on ukf (39K chunks, 6.4K edges), averaged over 10
runs with warm DuckDB cache.

| Query                       | Latency | Notes                            |
| --------------------------- | ------: | -------------------------------- |
| `get_chunks` (full, 39K)    |   262ms | Full deserialization             |
| `get_chunks(kind=FUNCTION)` |    18ms | Filtered                         |
| `search_by_name`            |    16ms | 12K total hits across 9 patterns |
| `search_fulltext` (BM25)    |    20ms | 80 hits across 8 queries         |
| `get_edges` (6.4K)          |    16ms | Full edge scan                   |
| `count_orphan_chunks`       |     1ms | Aggregate only                   |

**Key findings:**

- **Full `get_chunks` is slow** at 262ms for 39K chunks due to
  deserialising every row into a `Chunk` pydantic model. This is
  called during edge inference. Optimization: infer edges in SQL
  or use lazy iteration.
- **Search queries are fast** — sub-20ms for both BM25 and name
  search. More than sufficient for interactive use.

## Peak memory (RSS)

| Repo     |  Peak RSS | Delta from baseline |
| -------- | --------: | ------------------: |
| rbtr     |   236 MiB |             +94 MiB |
| typeshed |   833 MiB |            +460 MiB |
| ukf      | 1,006 MiB |            +660 MiB |

**Main contributors:**

1. **`all_chunks` list** — all extracted chunks are held in memory
   for batch DB insert and edge inference. 39K `Chunk` pydantic
   objects with string content is the primary allocation.
2. **PyArrow table construction** — `chunks_to_table()` creates
   columnar arrays from the chunk list, briefly doubling memory.
3. **DuckDB connection buffers** — query result buffers.
4. **Git blob reads** — `list_files()` materialises all file
   entries at once (generator → `list()`).

## DB size on disk

| Repo     | DB size | Chunks | Bytes/chunk |
| -------- | ------: | -----: | ----------: |
| rbtr     | 3.8 MiB |  2,264 |     1.7 KiB |
| typeshed |  28 MiB | 64,074 |     0.4 KiB |
| ukf      |  57 MiB | 39,287 |     1.5 KiB |

These are without embeddings. With embeddings (1024 × float32 =
4 KiB/chunk), ukf would add ~153 MiB of vector data.

## Embedding throughput (separate measurement)

Measured with bge-m3-Q4_K_M (418 MB GGUF):

| Metric                    |    Value | Notes            |
| ------------------------- | -------: | ---------------- |
| Model cold load           |    0.76s | GGUF from disk   |
| Short text (~50 tokens)   |     35ms | Metal GPU        |
| Medium text (~200 tokens) |    160ms | Metal GPU        |
| Real code chunks (mixed)  | 93ms avg | 200-chunk sample |

Throughput is dominated by per-call inference cost (~17 ms
baseline + ~0.7 ms/token). DB writes are negligible after
the batch-UPDATE optimization (10s for 39K chunks).

**Projected total embedding time (inference + DB writes):**

| Repo     | Chunks | Metal (est.) |
| -------- | -----: | -----------: |
| rbtr     |  2,264 |     ~3.5 min |
| typeshed | 64,074 |      ~99 min |
| ukf      | 39,287 |      ~61 min |

Embedding remains the dominant cost. The structural index
(parsing, chunking, edges) completes in under 4s for large
repos.

## Remaining optimization opportunities

Ranked by expected impact:

1. **Stream chunks to DB instead of accumulating in memory.**
   Insert chunks per-file instead of batch-all-then-insert.
   Edge inference can read chunks back from DB (already done
   for the `skipped_files > 0` path). Would cut peak memory
   roughly in half for large repos.

2. **Edge inference in SQL.** Instead of fetching all chunks into
   Python for import/test/doc edge inference, push the matching
   logic into DuckDB queries. Avoids the 262ms+ `get_chunks`
   deserialization for large repos.

3. **Generator-based file listing.** `list_files()` already
   returns a generator, but `build_index` calls `list()` on it
   to get total count for progress reporting. Could use a two-
   pass approach: count first (lightweight), then iterate.

## Running the benchmarks

```bash
# Quick benchmark (no embedding, no profiler dependency)
just bench                            # current repo, HEAD
just bench -- /path/to/repo main      # custom repo + ref
just bench -- . main feature          # with incremental update

# Profiling with scalene (line-level CPU + memory, needs debug group)
just bench-scalene -- /path/to/repo   # comprehensive bench under scalene
just scalene-view                     # view last profile in browser
```

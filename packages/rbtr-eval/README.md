# rbtr-eval

Search-quality evaluation harness for [rbtr](../rbtr).

## What it does

Measures whether docstring-derived natural-language queries
retrieve their owning symbol from the `rbtr` index. Two
jobs:

1. **Quantify what docstrings buy you.** Every repo is
   indexed twice — once in the production `full` variant
   (docstrings kept) and once in `stripped` (docstring
   bytes deleted). Both live in one shared `RBTR_HOME`;
   `rbtr search --variant {full,stripped}` picks which one
   to query. The Hit@k / MRR delta between variants is the
   cost of throwing away docstrings.
2. **Tune the search fusion weights `(alpha, beta, gamma)`.**
   Grid-searches over `[0, 1]^3` at a configurable step and
   reports the best triple by mean reciprocal rank. `tune`
   only reports; the operator decides whether to copy the
   weights into rbtr's source.

## Usage

Pipeline is driven by [DVC](https://dvc.org). From this
package directory:

```bash
uv run dvc repro            # run every stage end-to-end
uv run dvc repro tune       # also (or only) run the tune stage
uv run dvc repro extract    # rebuild from extract onwards
uv run dvc repro --dry      # show the plan without running
```

From the workspace root: `just eval`.

### Stages

| Stage            | What it does                                     | Output                                                     |
| ---------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| `clone@<slug>`   | `git clone --depth 1` each repo                  | `data/repos/<slug>/`                                       |
| `extract@<slug>` | sample docstring-derived queries from one repo   | `data/per-repo/<slug>.queries.parquet` + `.header.parquet` |
| `index`          | build `full` and `stripped` indexes sequentially | `data/home/`                                               |
| `measure`        | replay every query in every variant; aggregate   | `data/BENCHMARKS.md`, `data/metrics.json`                  |
| `tune`           | grid-search fusion weights against `full` index  | `data/tuned-params.json`                                   |

Indexing is sequential because only one embedding model may be
loaded at a time and DuckDB only tolerates one writer against
the shared home. `measure` and `tune` each open one warm
daemon for the duration of the stage (serves every search on
the same loaded model).

### Data layout

```text
data/
├── repos/                                # cloned benchmark repos
├── per-repo/
│   ├── <slug>.queries.parquet            # one row per sampled query (QueryRow schema)
│   └── <slug>.header.parquet             # one-row repo metadata (RepoHeader schema)
├── home/                                 # shared RBTR_HOME: both variants in one DuckDB
├── BENCHMARKS.md                         # report
├── metrics.json                          # DVC metrics
└── tuned-params.json                     # tuning suggestion
```

### Iterating on the harness

While developing, narrow `vars.repos` in `dvc.yaml` to one
entry (rbtr self) and lower `sample_cap`:

```yaml
vars:
  - sample_cap: 10
  - grid_step: 0.5
  - repos:
      - slug: rbtr__rbtr
        url: ../..
```

Restore the full four-repo list before the benchmark run.
End-to-end on the trimmed config takes ~30 seconds; the full
run with `sample_cap=300` takes about an hour (dominated by
indexing).

## Design

### Boundaries

- **extract.py** — the only module that imports rbtr's
  tree-sitter machinery. Walks one repo, builds a polars
  frame of documented symbols via `extract_doc_spans`,
  stratifies-samples by slug, writes two parquet files.
- **measure.py**, **tune.py** — open a typed `DaemonClient`
  against the shared home; every search is a
  `SearchRequest` in, a `SearchResponse` out. No
  subprocess parsing.
- **index_stage.py** — runs `python -m rbtr --no-daemon
  index` per `(repo, variant)`. No daemon; rbtr holds the
  DuckDB lock exclusively during each build.
- **rbtr_cli.py** — one place for every `python -m rbtr`
  shell-out and for the `daemon_session` context manager
  that wraps `DaemonClient`.

### Data

Every data frame boundary in `measure.py` / `tune.py` is a
dataframely `Schema`. Schemas live in
[`schemas.py`](src/rbtr_eval/schemas.py):

| Schema                  | Represents                                                      |
| ----------------------- | --------------------------------------------------------------- |
| `QueryRow`              | one sampled query (persisted as `<slug>.queries.parquet`)       |
| `RepoHeader`            | one repo's metadata (persisted as `<slug>.header.parquet`)      |
| `SearchBatch`           | raw `_run_searches` output: hits as `list[struct]`, pre-ranking |
| `SearchOutcome`         | per-(slug, variant, query) outcome: rank + latency + top-1 hit  |
| `Metrics`               | per-(slug, variant) metrics + `__all__` rollup                  |
| `MissCandidate`         | pivoted (full vs stripped) rows for the misses appendix         |
| `MetricsFile`           | on-disk `metrics.json` shape                                    |
| `WeightedSearchBatch`   | raw `_run_weight_trials` output                                 |
| `WeightedSearchOutcome` | per-(slug, label, triple, query) ranked outcome for tune        |
| `TuneReport`            | on-disk `tuned-params.json` shape                               |

Every function that takes or returns a frame annotates with
`dy.DataFrame[Schema]` and validates at the boundary via
`.pipe(Schema.validate, cast=True)`.

### Aggregation

All stats are polars expressions over typed frames. Hit@k:
`(pl.col("rank") <= k).fill_null(False).mean()`. MRR:
`pl.when(rank.is_null()).then(0.0).otherwise(1.0 / rank).mean()`.
Latency percentiles: `pl.col("latency_ms").quantile(0.5)`.
Ranking: `explode("hits") → int_range().over(outcome_keys) →
filter on target match → group_by().agg(min) → left-join back`.

No hand-rolled statistics, no DuckDB SQL strings in rbtr-eval.

### Markdown rendering

Tables in `BENCHMARKS.md` come from polars directly via
`pl.Config(tbl_formatting="MARKDOWN")` + `str(df)`. Each
table is a display-frame projection from its source schema;
columns, widths, and alignment are polars's job. The jinja
template inlines the pre-rendered strings.

Cosmetic markdown formatting (rumdl-style column alignment,
blank-line density) is left to `just lint-md` / CI; the
measure stage emits whatever polars and jinja produce.

## Tests

Pure-function tests only (`grid_triples`, `first_sentence`).
The polars pipeline is covered by an integration test that
feeds a synthetic `SearchBatch` through `_score_outcomes →
_aggregate → _select_misses → _render_report` against an
in-memory expected output.

Run: `just test-eval`.

## References

- Goal, pipeline, and findings history: [`todo/TODO-rbtr-eval.md`][todo]
- AGENTS rules that govern this package: [`AGENTS.md`][agents]

[todo]: ../../todo/TODO-rbtr-eval.md
[agents]: ../../AGENTS.md

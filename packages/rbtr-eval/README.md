# rbtr-eval

Search-quality evaluation harness for [rbtr](../rbtr).

## What it does

Measures whether docstring-derived natural-language queries
retrieve their owning symbol from the `rbtr` index. Used to:

1. Quantify the contribution of docstrings to search recall by
   indexing each repo twice — once with the production default
   (docstrings kept) and once with `rbtr index
--strip-docstrings`.
2. Tune the search fusion weights `(alpha, beta, gamma)` against
   the same query labels.

## Usage

The pipeline is driven by [DVC](https://dvc.org). From this
package directory:

```bash
uv run dvc repro            # run all stages: clone -> extract -> merge -> measure
uv run dvc repro tune       # also (or only) run the tune stage
uv run dvc repro extract    # rebuild from the per-repo extract step onwards
uv run dvc repro --dry      # show the plan without running
```

From the workspace root:

```bash
just eval                   # cd into this package and run dvc repro
```

### Stages

| Stage            | What it does                                   | Output                                    |
| ---------------- | ---------------------------------------------- | ----------------------------------------- |
| `clone@<slug>`   | `git clone --depth 1` each repo                | `data/repos/<slug>/`                      |
| `extract@<slug>` | sample docstring-derived queries from one repo | `data/per-repo/<slug>.jsonl`              |
| `merge-dataset`  | concatenate per-repo files into one dataset    | `data/dataset.jsonl`                      |
| `measure`        | build both indexes per repo, replay queries    | `data/BENCHMARKS.md`, `data/metrics.json` |
| `tune`           | grid-search fusion weights against the dataset | `data/tuned-params.json`                  |

`tune` only reports the suggested weights; the operator decides
whether to copy them into rbtr's source.

### Data layout

```text
data/
├── repos/                # cloned benchmark repos (gitignored, dvc-tracked)
├── homes/                # per-(repo, mode) RBTR_HOME trees (gitignored, dvc-tracked)
├── per-repo/             # per-repo gather output (gitignored, dvc-cached)
├── dataset.jsonl         # merged dataset (gitignored, dvc-cached)
├── BENCHMARKS.md         # report (git-tracked)
├── metrics.json          # aggregate metrics (git-tracked)
└── tuned-params.json     # tuning suggestion (git-tracked)
```

## Iterating on the harness

While developing the Python tools, narrow `vars.repos` in
`dvc.yaml` to one entry (rbtr self) and lower `sample_cap` to
something tiny (e.g. 20). Restore the full four-repo list
before the final benchmark run.

## Design

The Python tools are deliberately narrow:

- `extract.py` — the only module that imports from `rbtr`,
  for docstring identification (uses
  `rbtr.index.treesitter.extract_doc_spans`).
- `measure.py`, `tune.py` — pure subprocess; they shell out
  to `rbtr index` and `rbtr --json search` and parse the JSON.

Cloning and orchestration live in `dvc.yaml`. No shell scripts
checked in; DVC's `cmd:` strings are bare commands.

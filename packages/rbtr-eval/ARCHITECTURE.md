# rbtr-eval architecture

## Layers

Each subcommand has three layers.

- **I/O.** The `*Cmd.cli_cmd` method and the named readers it calls
  read parquet and the index, call the daemon or the LLM, and write
  the report, its charts and `metrics.json`. Nothing else does.
- **Statistics.** Pure functions from typed frames to one typed frame
  per table, holding raw numbers.
- **Rendering.** Pure functions from those frames, passed as
  parameters, to markdown and chart specs. Each table is a named local
  built from display expressions; the template receives only names.

## Tables

Ranking figures (Hit@k, MRR, NDCG) do not add up across groups, so
each breakdown is computed from the outcomes: a long frame of
`dimension, value, figures`, one row per value of one `Dimension`.
The figures over every query are a frame of their own.

Counts add up, so the profile, expansion and paraphrase reports
compute one joint count, and each of their tables sums it.

## `metrics.json`

Nested objects: arm, then `all` or a query kind, then figure. DVC
reads metrics only from objects, and `dvc metrics diff` then names
each figure by its path, as `none.concept.mrr`.

## Rejected

- **One stacked rollup, with `__all__` in the columns a row does not
  group by.** Every reader had to decode the sentinel to find a level,
  which made it a small language on top of polars.
- **`metrics.json` as a list of rows.** `dvc metrics show` prints no
  value from a list (checked with DVC 3.67).
- **A dataclass or pydantic model bundling a command's frames.** Each
  frame is validated where it is built; a container adds nothing that
  plain parameters lack.
- **Tables computed inside the `render_str` call.** It tangled
  computing, formatting and templating, and tests could check the
  figures only by searching the markdown.

# rbtr search-quality benchmark

Hit@k / MRR / NDCG@10 for queries against the rbtr index. See
`packages/rbtr-eval/README.md` for methodology.

## Run

Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.

| field         | value                                     |
| ------------- | ----------------------------------------- |
| seed          | 0                                         |
| sample target | 10 per (repo, language, kind, provenance) |
| total queries | 3618                                      |
| elapsed       | 56873 s                                   |

## Headline metrics

No-expansion baseline (arm `none`) per repo. The expansion
ablation is in the section below.

| repo                 | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| -------------------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| **all repos**        | 3618 | 50.4% | 68.6% | 81.0%  | 0.608 | 0.657   | 1           | 19.0%     |
| `anthropics__skills` | 734  | 63.6% | 82.4% | 92.9%  | 0.741 | 0.787   | 1           | 7.1%      |
| `astral-sh__uv`      | 895  | 48.3% | 66.7% | 79.4%  | 0.589 | 0.639   | 1           | 20.6%     |
| `badlogic__pi-mono`  | 660  | 50.6% | 67.4% | 81.1%  | 0.606 | 0.655   | 1           | 18.9%     |
| `django__django`     | 766  | 38.9% | 54.2% | 65.7%  | 0.476 | 0.52    | 1           | 34.3%     |
| `rbtr__rbtr`         | 563  | 51.7% | 74.4% | 88.6%  | 0.644 | 0.704   | 1           | 11.4%     |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 3618 | 48.1% | 66.3% | 79.3%  | 0.586 | 0.636   |
| keywords | **all**      | 3618 | 48.8% | 66.6% | 79.4%  | 0.591 | 0.64    |
| none     | **all**      | 3618 | 50.4% | 68.6% | 81.0%  | 0.608 | 0.657   |
| variants | **all**      | 3618 | 49.8% | 68.7% | 80.8%  | 0.605 | 0.655   |
| both     | `code`       | 540  | 74.3% | 90.2% | 94.1%  | 0.824 | 0.853   |
| keywords | `code`       | 540  | 74.4% | 90.2% | 94.1%  | 0.825 | 0.854   |
| none     | `code`       | 540  | 77.0% | 90.6% | 94.8%  | 0.841 | 0.868   |
| variants | `code`       | 540  | 76.7% | 90.6% | 94.8%  | 0.839 | 0.866   |
| both     | `concept`    | 1565 | 36.7% | 58.5% | 74.6%  | 0.494 | 0.555   |
| keywords | `concept`    | 1565 | 37.8% | 59.2% | 74.9%  | 0.501 | 0.561   |
| none     | `concept`    | 1565 | 38.4% | 59.4% | 74.6%  | 0.504 | 0.563   |
| variants | `concept`    | 1565 | 37.6% | 59.4% | 74.2%  | 0.501 | 0.559   |
| both     | `identifier` | 1513 | 50.5% | 65.9% | 78.8%  | 0.597 | 0.643   |
| keywords | `identifier` | 1513 | 51.0% | 65.9% | 78.7%  | 0.6   | 0.645   |
| none     | `identifier` | 1513 | 53.2% | 70.3% | 82.6%  | 0.631 | 0.679   |
| variants | `identifier` | 1513 | 52.9% | 70.5% | 82.6%  | 0.63  | 0.678   |

![MRR by repo](mrr_by_repo.png)

## Per-kind breakdown

Retrieval quality for each target chunk kind (`symbol_kind`),
aggregated across repos, languages and provenances.

| symbol_kind   | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | not found |
| ------------- | --- | ----- | ----- | ------ | ----- | ------- | --------- |
| `function`    | 626 | 72.7% | 87.9% | 93.9%  | 0.807 | 0.839   | 6.1%      |
| `class`       | 634 | 68.3% | 87.2% | 93.1%  | 0.781 | 0.818   | 6.9%      |
| `method`      | 424 | 55.2% | 80.0% | 92.7%  | 0.687 | 0.746   | 7.3%      |
| `variable`    | 722 | 52.1% | 72.9% | 88.8%  | 0.643 | 0.702   | 11.2%     |
| `comment`     | 330 | 37.9% | 60.9% | 77.0%  | 0.513 | 0.575   | 23.0%     |
| `config_key`  | 338 | 24.3% | 38.2% | 49.7%  | 0.325 | 0.366   | 50.3%     |
| `doc_section` | 270 | 24.4% | 34.1% | 49.3%  | 0.309 | 0.352   | 50.7%     |
| `raw_chunk`   | 274 | 18.6% | 33.2% | 59.5%  | 0.295 | 0.365   | 40.5%     |

![MRR by target kind](mrr_by_kind.png)

## Target × request shape

MRR for each target kind sliced by request shape (`query_kind`) —
which kinds search finds well, and via which kind of query.

| symbol_kind   | concept | identifier | code  |
| ------------- | ------- | ---------- | ----- |
| `class`       | 0.702   | 0.796      | 0.896 |
| `comment`     | 0.447   | 0.606      | 0.333 |
| `config_key`  | 0.222   | 0.386      | 0.614 |
| `doc_section` | 0.207   | 0.361      | 0.639 |
| `function`    | 0.725   | 0.816      | 0.941 |
| `method`      | 0.623   | 0.639      | 0.896 |
| `raw_chunk`   | 0.248   | 0.312      | 0.522 |
| `variable`    | 0.522   | 0.712      | 0.797 |

## Search latency

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 3151 ms    | 7690 ms    |
| `anthropics__skills` | 2977 ms    | 7014 ms    |
| `astral-sh__uv`      | 3360 ms    | 8939 ms    |
| `badlogic__pi-mono`  | 2812 ms    | 7101 ms    |
| `django__django`     | 3544 ms    | 7293 ms    |
| `rbtr__rbtr`         | 2996 ms    | 7065 ms    |

## Per-language breakdown

Aggregated across all repos for each language present in the sample.

| language     | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ------------ | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| ``           | 107 | 23.4% | 44.9% | 70.1%  | 0.376 | 0.455   | 2           | 29.9%     |
| `bash`       | 233 | 72.1% | 88.0% | 97.0%  | 0.807 | 0.847   | 1           | 3.0%      |
| `css`        | 355 | 61.1% | 80.6% | 93.5%  | 0.726 | 0.777   | 1           | 6.5%      |
| `html`       | 58  | 24.1% | 34.5% | 44.8%  | 0.306 | 0.34    | 1           | 55.2%     |
| `javascript` | 431 | 63.6% | 79.6% | 86.8%  | 0.721 | 0.757   | 1           | 13.2%     |
| `json`       | 200 | 15.5% | 29.5% | 43.0%  | 0.239 | 0.284   | 2           | 57.0%     |
| `markdown`   | 281 | 24.9% | 35.6% | 57.7%  | 0.33  | 0.387   | 2           | 42.3%     |
| `python`     | 898 | 59.4% | 78.1% | 87.4%  | 0.694 | 0.739   | 1           | 12.6%     |
| `rst`        | 40  | 12.5% | 25.0% | 37.5%  | 0.202 | 0.244   | 2           | 62.5%     |
| `rust`       | 248 | 46.8% | 72.2% | 86.3%  | 0.605 | 0.668   | 1           | 13.7%     |
| `sql`        | 80  | 50.0% | 76.2% | 93.8%  | 0.648 | 0.718   | 1           | 6.2%      |
| `toml`       | 74  | 41.9% | 47.3% | 55.4%  | 0.455 | 0.478   | 1           | 44.6%     |
| `typescript` | 521 | 52.2% | 75.0% | 88.7%  | 0.652 | 0.709   | 1           | 11.3%     |
| `yaml`       | 92  | 28.3% | 46.7% | 62.0%  | 0.397 | 0.451   | 2           | 38.0%     |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 913  | 64.7% | 82.1% | 89.0%  | 0.74  | 0.777   | 1           | 11.0%     |
| `concept`   | 1523 | 38.0% | 59.2% | 74.3%  | 0.501 | 0.56    | 1           | 25.7%     |
| `docstring` | 389  | 71.2% | 85.6% | 90.0%  | 0.788 | 0.816   | 1           | 10.0%     |
| `name`      | 793  | 47.3% | 62.7% | 80.1%  | 0.572 | 0.627   | 1           | 19.9%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n     | MRR   |
| --------- | ----- | ----- |
| full      | 14288 | 74.6% |
| truncated | 184   | 55.4% |

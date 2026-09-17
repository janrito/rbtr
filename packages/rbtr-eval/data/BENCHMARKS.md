# rbtr search-quality benchmark

Hit@k / MRR / NDCG@10 for queries against the rbtr index. See
`packages/rbtr-eval/README.md` for methodology.

## Run

Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.

| field         | value                                     |
| ------------- | ----------------------------------------- |
| seed          | 0                                         |
| sample target | 10 per (repo, language, kind, provenance) |
| total queries | 3641                                      |
| elapsed       | 110140 s                                  |

## Headline metrics

No-expansion baseline (arm `none`) per repo. The expansion
ablation is in the section below.

| repo                 | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| -------------------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| **all repos**        | 3641 | 50.7% | 69.0% | 80.3%  | 0.61  | 0.657   | 1           | 19.7%     |
| `anthropics__skills` | 738  | 62.1% | 82.1% | 90.5%  | 0.727 | 0.771   | 1           | 9.5%      |
| `astral-sh__uv`      | 916  | 47.3% | 65.5% | 77.1%  | 0.575 | 0.623   | 1           | 22.9%     |
| `badlogic__pi-mono`  | 647  | 50.7% | 69.7% | 80.8%  | 0.612 | 0.66    | 1           | 19.2%     |
| `django__django`     | 751  | 43.1% | 57.9% | 69.0%  | 0.516 | 0.558   | 1           | 31.0%     |
| `rbtr__rbtr`         | 589  | 51.4% | 71.5% | 86.4%  | 0.634 | 0.69    | 1           | 13.6%     |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 3641 | 48.7% | 66.9% | 78.3%  | 0.59  | 0.637   |
| keywords | **all**      | 3641 | 49.1% | 66.9% | 78.4%  | 0.592 | 0.639   |
| none     | **all**      | 3641 | 50.7% | 69.0% | 80.3%  | 0.61  | 0.657   |
| variants | **all**      | 3641 | 50.5% | 68.5% | 80.0%  | 0.608 | 0.655   |
| both     | `code`       | 580  | 72.8% | 87.4% | 91.6%  | 0.803 | 0.831   |
| keywords | `code`       | 580  | 72.9% | 87.4% | 91.4%  | 0.804 | 0.832   |
| none     | `code`       | 580  | 75.0% | 88.3% | 91.6%  | 0.818 | 0.842   |
| variants | `code`       | 580  | 75.0% | 88.3% | 91.7%  | 0.818 | 0.843   |
| both     | `concept`    | 1623 | 35.6% | 57.1% | 72.6%  | 0.479 | 0.539   |
| keywords | `concept`    | 1623 | 36.3% | 57.5% | 73.0%  | 0.484 | 0.544   |
| none     | `concept`    | 1623 | 35.9% | 57.2% | 73.1%  | 0.481 | 0.542   |
| variants | `concept`    | 1623 | 35.3% | 56.1% | 72.3%  | 0.475 | 0.535   |
| both     | `identifier` | 1438 | 53.8% | 69.7% | 79.5%  | 0.629 | 0.67    |
| keywords | `identifier` | 1438 | 53.9% | 69.3% | 79.2%  | 0.629 | 0.669   |
| none     | `identifier` | 1438 | 57.6% | 74.5% | 83.9%  | 0.671 | 0.712   |
| variants | `identifier` | 1438 | 57.9% | 74.5% | 83.9%  | 0.672 | 0.713   |

![MRR by repo](mrr_by_repo.png)

## Per-kind breakdown

Retrieval quality for each target chunk kind (`symbol_kind`),
aggregated across repos, languages and provenances.

| symbol_kind   | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | not found |
| ------------- | --- | ----- | ----- | ------ | ----- | ------- | --------- |
| `function`    | 643 | 67.0% | 85.4% | 93.5%  | 0.766 | 0.808   | 6.5%      |
| `class`       | 643 | 66.7% | 82.4% | 92.2%  | 0.757 | 0.798   | 7.8%      |
| `variable`    | 759 | 54.0% | 75.2% | 86.7%  | 0.658 | 0.709   | 13.3%     |
| `method`      | 433 | 51.0% | 76.2% | 91.2%  | 0.651 | 0.715   | 8.8%      |
| `comment`     | 356 | 36.0% | 59.0% | 74.4%  | 0.493 | 0.554   | 25.6%     |
| `raw_chunk`   | 60  | 26.7% | 56.7% | 66.7%  | 0.42  | 0.481   | 33.3%     |
| `doc_section` | 403 | 31.0% | 40.0% | 51.1%  | 0.367 | 0.401   | 48.9%     |
| `config_key`  | 344 | 25.0% | 37.2% | 48.3%  | 0.323 | 0.361   | 51.7%     |

![MRR by target kind](mrr_by_kind.png)

## Target × request shape

MRR for each target kind sliced by request shape (`query_kind`) —
which kinds search finds well, and via which kind of query.

| symbol_kind   | concept | identifier | code  |
| ------------- | ------- | ---------- | ----- |
| `class`       | 0.654   | 0.803      | 0.874 |
| `comment`     | 0.404   | 0.625      | 0.215 |
| `config_key`  | 0.233   | 0.388      | 0.495 |
| `doc_section` | 0.205   | 0.487      | 0.687 |
| `function`    | 0.656   | 0.804      | 0.912 |
| `method`      | 0.552   | 0.621      | 0.901 |
| `raw_chunk`   | 0.361   | 0.433      | 0.52  |
| `variable`    | 0.522   | 0.741      | 0.825 |

## Search latency

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 6791 ms    | 15427 ms   |
| `anthropics__skills` | 6123 ms    | 16980 ms   |
| `astral-sh__uv`      | 6809 ms    | 15999 ms   |
| `badlogic__pi-mono`  | 5969 ms    | 12208 ms   |
| `django__django`     | 7159 ms    | 13784 ms   |
| `rbtr__rbtr`         | 7173 ms    | 16063 ms   |

## Per-language breakdown

Aggregated across all repos for each language present in the sample.

| language     | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ------------ | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `bash`       | 392 | 62.0% | 79.1% | 88.8%  | 0.717 | 0.759   | 1           | 11.2%     |
| `css`        | 349 | 59.9% | 81.1% | 92.8%  | 0.716 | 0.768   | 1           | 7.2%      |
| `html`       | 38  | 28.9% | 34.2% | 47.4%  | 0.33  | 0.363   | 1           | 52.6%     |
| `javascript` | 430 | 62.6% | 79.8% | 84.9%  | 0.711 | 0.745   | 1           | 15.1%     |
| `json`       | 199 | 18.1% | 30.7% | 44.2%  | 0.259 | 0.302   | 2           | 55.8%     |
| `markdown`   | 199 | 27.1% | 39.2% | 52.3%  | 0.344 | 0.387   | 1           | 47.7%     |
| `plaintext`  | 60  | 26.7% | 56.7% | 66.7%  | 0.42  | 0.481   | 2           | 33.3%     |
| `python`     | 890 | 59.8% | 78.3% | 88.8%  | 0.701 | 0.747   | 1           | 11.2%     |
| `rst`        | 70  | 17.1% | 18.6% | 24.3%  | 0.187 | 0.2     | 1           | 75.7%     |
| `rust`       | 252 | 44.0% | 66.7% | 84.9%  | 0.569 | 0.636   | 1           | 15.1%     |
| `sql`        | 80  | 51.2% | 72.5% | 92.5%  | 0.647 | 0.715   | 1           | 7.5%      |
| `toml`       | 72  | 38.9% | 48.6% | 52.8%  | 0.438 | 0.46    | 1           | 47.2%     |
| `typescript` | 517 | 49.7% | 72.9% | 87.0%  | 0.632 | 0.69    | 1           | 13.0%     |
| `yaml`       | 93  | 29.0% | 46.2% | 58.1%  | 0.389 | 0.435   | 1           | 41.9%     |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 967  | 66.3% | 82.8% | 88.9%  | 0.751 | 0.786   | 1           | 11.1%     |
| `concept`   | 1576 | 35.2% | 56.5% | 72.4%  | 0.474 | 0.535   | 2           | 27.6%     |
| `docstring` | 389  | 71.0% | 85.9% | 89.7%  | 0.786 | 0.814   | 1           | 10.3%     |
| `name`      | 709  | 52.9% | 68.7% | 81.0%  | 0.621 | 0.667   | 1           | 19.0%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n     | MRR   |
| --------- | ----- | ----- |
| full      | 14372 | 75.7% |
| truncated | 192   | 64.8% |

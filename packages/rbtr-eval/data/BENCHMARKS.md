# rbtr search-quality benchmark

Hit@k / MRR / NDCG@10 for queries against the rbtr index. See
`packages/rbtr-eval/README.md` for methodology.

## Run

Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.

| field         | value                               |
| ------------- | ----------------------------------- |
| seed          | 0                                   |
| sample target | 25 per (repo, language, provenance) |
| total queries | 2586                                |
| elapsed       | 43412 s                             |

## Repos

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 1616    | 125             |
| `astral-sh__uv`      | `cfe5277bc422` | 20109   | 350             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 13498   | 292             |
| `django__django`     | `e78a46a8fb29` | 60582   | 450             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 4748    | 247             |

## Headline metrics

No-expansion baseline (arm `none`) per repo. The expansion
ablation is in the section below.

| repo                 | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| -------------------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| **all repos**        | 2586 | 41.2% | 55.7% | 64.5%  | 0.492 | 0.529   | 1           | 35.5%     |
| `anthropics__skills` | 224  | 48.2% | 76.3% | 90.2%  | 0.634 | 0.699   | 1           | 9.8%      |
| `astral-sh__uv`      | 626  | 35.1% | 47.0% | 58.0%  | 0.424 | 0.461   | 1           | 42.0%     |
| `badlogic__pi-mono`  | 505  | 50.9% | 65.0% | 71.1%  | 0.584 | 0.616   | 1           | 28.9%     |
| `django__django`     | 807  | 34.1% | 46.1% | 54.5%  | 0.408 | 0.441   | 1           | 45.5%     |
| `rbtr__rbtr`         | 424  | 48.6% | 64.9% | 71.7%  | 0.57  | 0.606   | 1           | 28.3%     |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 2586 | 40.9% | 55.5% | 64.7%  | 0.492 | 0.529   |
| keywords | **all**      | 2586 | 40.8% | 55.3% | 64.8%  | 0.491 | 0.529   |
| none     | **all**      | 2586 | 41.2% | 55.7% | 64.5%  | 0.492 | 0.529   |
| variants | **all**      | 2586 | 41.0% | 55.7% | 64.2%  | 0.491 | 0.527   |
| both     | `code`       | 402  | 66.7% | 83.3% | 91.0%  | 0.758 | 0.795   |
| keywords | `code`       | 402  | 66.2% | 82.6% | 90.5%  | 0.754 | 0.791   |
| none     | `code`       | 402  | 69.2% | 84.8% | 92.0%  | 0.778 | 0.813   |
| variants | `code`       | 402  | 69.4% | 85.1% | 92.0%  | 0.779 | 0.814   |
| both     | `concept`    | 1155 | 26.1% | 42.1% | 53.0%  | 0.353 | 0.396   |
| keywords | `concept`    | 1155 | 26.3% | 41.7% | 53.2%  | 0.354 | 0.397   |
| none     | `concept`    | 1155 | 25.2% | 39.9% | 50.4%  | 0.336 | 0.377   |
| variants | `concept`    | 1155 | 24.7% | 40.0% | 49.7%  | 0.332 | 0.372   |
| both     | `identifier` | 1029 | 47.3% | 59.8% | 67.5%  | 0.543 | 0.575   |
| keywords | `identifier` | 1029 | 47.1% | 59.8% | 67.7%  | 0.543 | 0.575   |
| none     | `identifier` | 1029 | 48.3% | 62.0% | 69.6%  | 0.556 | 0.59    |
| variants | `identifier` | 1029 | 48.3% | 61.9% | 69.7%  | 0.556 | 0.59    |

![MRR by repo](mrr_by_repo.png)

## Search latency

Shared index home size: **2.6 GiB**.

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 3325 ms    | 8301 ms    |
| `anthropics__skills` | 4533 ms    | 10238 ms   |
| `astral-sh__uv`      | 3115 ms    | 9564 ms    |
| `badlogic__pi-mono`  | 3131 ms    | 7754 ms    |
| `django__django`     | 3714 ms    | 7529 ms    |
| `rbtr__rbtr`         | 2880 ms    | 6843 ms    |

## Classification

Cross-tabulation of query provenance (where the query was sampled
from) against online `QueryKind` (how `classify_query` routes it
for expansion).

| provenance  | CONCEPT | IDENTIFIER | CODE  | n    |
| ----------- | ------- | ---------- | ----- | ---- |
| `body`      | 2.2%    | 39.2%      | 58.6% | 2500 |
| `concept`   | 99.2%   | 0.8%       | null  | 4488 |
| `docstring` | 1.9%    | 91.1%      | 7.0%  | 856  |
| `name`      | 3.8%    | 92.8%      | 3.4%  | 2500 |

## Per-language breakdown

Aggregated across all repos for each language present in the sample.

| language     | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ------------ | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `css`        | 150 | 74.7% | 84.0% | 88.0%  | 0.8   | 0.82    | 1           | 12.0%     |
| `html`       | 95  | 29.5% | 42.1% | 50.5%  | 0.368 | 0.401   | 1           | 49.5%     |
| `javascript` | 238 | 68.9% | 82.4% | 83.6%  | 0.753 | 0.774   | 1           | 16.4%     |
| `json`       | 392 | 9.7%  | 17.6% | 25.5%  | 0.146 | 0.172   | 2           | 74.5%     |
| `markdown`   | 490 | 23.5% | 39.0% | 52.0%  | 0.326 | 0.373   | 2           | 48.0%     |
| `python`     | 493 | 58.4% | 82.6% | 93.3%  | 0.709 | 0.764   | 1           | 6.7%      |
| `rst`        | 98  | 13.3% | 16.3% | 33.7%  | 0.177 | 0.214   | 4           | 66.3%     |
| `rust`       | 144 | 54.2% | 70.8% | 83.3%  | 0.645 | 0.691   | 1           | 16.7%     |
| `toml`       | 84  | 26.2% | 36.9% | 38.1%  | 0.311 | 0.328   | 1           | 61.9%     |
| `typescript` | 235 | 75.7% | 94.5% | 98.7%  | 0.848 | 0.883   | 1           | 1.3%      |
| `yaml`       | 167 | 18.0% | 24.0% | 34.1%  | 0.22  | 0.249   | 1           | 65.9%     |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 625  | 57.1% | 72.6% | 82.1%  | 0.659 | 0.698   | 1           | 17.9%     |
| `concept`   | 1122 | 25.4% | 40.4% | 50.5%  | 0.339 | 0.379   | 1           | 49.5%     |
| `docstring` | 214  | 78.5% | 91.1% | 93.5%  | 0.847 | 0.869   | 1           | 6.5%      |
| `name`      | 625  | 41.0% | 54.1% | 62.1%  | 0.48  | 0.514   | 1           | 37.9%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n     | MRR   |
| --------- | ----- | ----- |
| full      | 10236 | 76.4% |
| truncated | 108   | 44.6% |

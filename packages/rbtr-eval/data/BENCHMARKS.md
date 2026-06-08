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
| elapsed       | 44544 s                             |

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
| **all repos**        | 2586 | 42.1% | 56.5% | 64.9%  | 0.5   | 0.536   | 1           | 35.1%     |
| `anthropics__skills` | 224  | 56.2% | 79.0% | 91.1%  | 0.686 | 0.741   | 1           | 8.9%      |
| `astral-sh__uv`      | 626  | 34.8% | 46.8% | 58.0%  | 0.422 | 0.459   | 1           | 42.0%     |
| `badlogic__pi-mono`  | 505  | 49.7% | 65.3% | 71.7%  | 0.579 | 0.613   | 1           | 28.3%     |
| `django__django`     | 807  | 33.1% | 46.7% | 54.8%  | 0.404 | 0.439   | 1           | 45.2%     |
| `rbtr__rbtr`         | 424  | 53.5% | 66.7% | 72.4%  | 0.606 | 0.635   | 1           | 27.6%     |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 2586 | 41.4% | 56.0% | 64.8%  | 0.495 | 0.532   |
| keywords | **all**      | 2586 | 41.4% | 56.4% | 64.8%  | 0.496 | 0.533   |
| none     | **all**      | 2586 | 42.1% | 56.5% | 64.9%  | 0.5   | 0.536   |
| variants | **all**      | 2586 | 42.3% | 56.4% | 64.8%  | 0.5   | 0.536   |
| both     | `code`       | 402  | 67.4% | 83.6% | 89.8%  | 0.759 | 0.793   |
| keywords | `code`       | 402  | 66.9% | 83.6% | 90.0%  | 0.757 | 0.793   |
| none     | `code`       | 402  | 69.9% | 84.6% | 91.5%  | 0.779 | 0.812   |
| variants | `code`       | 402  | 70.1% | 84.8% | 91.3%  | 0.779 | 0.812   |
| both     | `concept`    | 1155 | 27.0% | 42.9% | 54.4%  | 0.361 | 0.405   |
| keywords | `concept`    | 1155 | 27.1% | 43.5% | 54.1%  | 0.362 | 0.406   |
| none     | `concept`    | 1155 | 26.1% | 41.2% | 51.2%  | 0.346 | 0.386   |
| variants | `concept`    | 1155 | 26.1% | 41.2% | 51.1%  | 0.345 | 0.385   |
| both     | `identifier` | 1029 | 47.4% | 60.0% | 66.7%  | 0.542 | 0.572   |
| keywords | `identifier` | 1029 | 47.4% | 60.3% | 67.0%  | 0.543 | 0.574   |
| none     | `identifier` | 1029 | 49.3% | 62.6% | 69.9%  | 0.564 | 0.596   |
| variants | `identifier` | 1029 | 49.6% | 62.3% | 69.9%  | 0.565 | 0.597   |

![MRR by repo](mrr_by_repo.png)

## Search latency

Shared index home size: **1.6 GiB**.

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 3507 ms    | 8369 ms    |
| `anthropics__skills` | 4718 ms    | 10219 ms   |
| `astral-sh__uv`      | 3098 ms    | 9473 ms    |
| `badlogic__pi-mono`  | 3287 ms    | 8329 ms    |
| `django__django`     | 4008 ms    | 7698 ms    |
| `rbtr__rbtr`         | 2880 ms    | 6844 ms    |

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
| `css`        | 150 | 72.0% | 84.0% | 88.7%  | 0.787 | 0.812   | 1           | 11.3%     |
| `html`       | 95  | 26.3% | 40.0% | 50.5%  | 0.345 | 0.384   | 1           | 49.5%     |
| `javascript` | 238 | 68.9% | 83.2% | 84.5%  | 0.756 | 0.779   | 1           | 15.5%     |
| `json`       | 392 | 9.4%  | 17.3% | 26.0%  | 0.144 | 0.172   | 2           | 74.0%     |
| `markdown`   | 490 | 23.9% | 40.6% | 52.9%  | 0.334 | 0.381   | 2           | 47.1%     |
| `python`     | 493 | 62.9% | 84.2% | 93.1%  | 0.739 | 0.786   | 1           | 6.9%      |
| `rst`        | 98  | 13.3% | 19.4% | 34.7%  | 0.184 | 0.222   | 3           | 65.3%     |
| `rust`       | 144 | 54.2% | 70.8% | 83.3%  | 0.646 | 0.692   | 1           | 16.7%     |
| `toml`       | 84  | 27.4% | 36.9% | 38.1%  | 0.317 | 0.333   | 1           | 61.9%     |
| `typescript` | 235 | 77.9% | 95.3% | 99.1%  | 0.864 | 0.896   | 1           | 0.9%      |
| `yaml`       | 167 | 18.6% | 24.0% | 34.1%  | 0.222 | 0.25    | 1           | 65.9%     |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 625  | 57.1% | 73.1% | 81.9%  | 0.658 | 0.697   | 1           | 18.1%     |
| `concept`   | 1122 | 26.4% | 41.6% | 51.3%  | 0.349 | 0.389   | 1           | 48.7%     |
| `docstring` | 214  | 79.9% | 90.7% | 93.0%  | 0.853 | 0.872   | 1           | 7.0%      |
| `name`      | 625  | 42.4% | 54.7% | 62.6%  | 0.491 | 0.524   | 1           | 37.4%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n     | MRR   |
| --------- | ----- | ----- |
| full      | 10236 | 77.0% |
| truncated | 108   | 46.3% |

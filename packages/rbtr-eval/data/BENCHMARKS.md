# rbtr search-quality benchmark

Hit@k / MRR / NDCG@10 for queries against the rbtr index. See
`packages/rbtr-eval/README.md` for methodology.

## Run

Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.

| field         | value                               |
| ------------- | ----------------------------------- |
| seed          | 0                                   |
| sample target | 25 per (repo, language, provenance) |
| total queries | 2357                                |
| elapsed       | 38295 s                             |

## Repos

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 1913    | 264             |
| `astral-sh__uv`      | `cfe5277bc422` | 19007   | 200             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 9238    | 262             |
| `django__django`     | `e78a46a8fb29` | 59123   | 375             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 2192    | 263             |

## Headline metrics

No-expansion baseline (arm `none`) per repo. The expansion
ablation is in the section below.

| repo                 | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| -------------------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| **all repos**        | 2357 | 55.2% | 71.2% | 81.5%  | 0.644 | 0.686   | 1           | 18.5%     |
| `anthropics__skills` | 438  | 71.2% | 86.8% | 94.1%  | 0.796 | 0.832   | 1           | 5.9%      |
| `astral-sh__uv`      | 359  | 47.6% | 66.3% | 76.9%  | 0.581 | 0.627   | 1           | 23.1%     |
| `badlogic__pi-mono`  | 460  | 61.5% | 73.0% | 81.5%  | 0.684 | 0.716   | 1           | 18.5%     |
| `django__django`     | 672  | 45.4% | 57.7% | 70.1%  | 0.533 | 0.573   | 1           | 29.9%     |
| `rbtr__rbtr`         | 428  | 53.7% | 78.7% | 90.2%  | 0.675 | 0.731   | 1           | 9.8%      |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 2357 | 54.0% | 70.0% | 80.3%  | 0.632 | 0.674   |
| keywords | **all**      | 2357 | 54.4% | 70.5% | 80.6%  | 0.636 | 0.678   |
| none     | **all**      | 2357 | 55.2% | 71.2% | 81.5%  | 0.644 | 0.686   |
| variants | **all**      | 2357 | 54.9% | 70.9% | 81.0%  | 0.641 | 0.682   |
| both     | `code`       | 419  | 77.6% | 88.3% | 92.6%  | 0.835 | 0.857   |
| keywords | `code`       | 419  | 77.6% | 88.3% | 92.6%  | 0.835 | 0.858   |
| none     | `code`       | 419  | 78.0% | 88.3% | 92.6%  | 0.838 | 0.86    |
| variants | `code`       | 419  | 77.8% | 88.3% | 92.6%  | 0.836 | 0.859   |
| both     | `concept`    | 999  | 42.8% | 61.3% | 76.2%  | 0.538 | 0.592   |
| keywords | `concept`    | 999  | 43.3% | 62.0% | 76.7%  | 0.544 | 0.598   |
| none     | `concept`    | 999  | 43.2% | 61.3% | 75.6%  | 0.541 | 0.593   |
| variants | `concept`    | 999  | 42.7% | 60.8% | 74.9%  | 0.535 | 0.587   |
| both     | `identifier` | 939  | 55.3% | 71.2% | 79.2%  | 0.643 | 0.68    |
| keywords | `identifier` | 939  | 55.8% | 71.6% | 79.3%  | 0.646 | 0.682   |
| none     | `identifier` | 939  | 57.7% | 74.2% | 82.7%  | 0.668 | 0.707   |
| variants | `identifier` | 939  | 57.5% | 73.9% | 82.4%  | 0.667 | 0.705   |

![MRR by repo](mrr_by_repo.png)

## Search latency

Shared index home size: **1.9 GiB**.

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 3194 ms    | 8007 ms    |
| `anthropics__skills` | 3173 ms    | 7029 ms    |
| `astral-sh__uv`      | 3180 ms    | 9955 ms    |
| `badlogic__pi-mono`  | 2894 ms    | 7778 ms    |
| `django__django`     | 3558 ms    | 7347 ms    |
| `rbtr__rbtr`         | 3156 ms    | 7815 ms    |

## Classification

Cross-tabulation of query provenance (where the query was sampled
from) against online `QueryKind` (how `classify_query` routes it
for expansion).

| provenance  | CONCEPT | IDENTIFIER | CODE  | n    |
| ----------- | ------- | ---------- | ----- | ---- |
| `body`      | 1.5%    | 26.9%      | 71.6% | 2100 |
| `concept`   | 97.8%   | 2.1%       | 0.1%  | 3972 |
| `docstring` | 2.5%    | 91.1%      | 6.4%  | 1256 |
| `name`      | 2.3%    | 93.5%      | 4.2%  | 2100 |

## Per-language breakdown

Aggregated across all repos for each language present in the sample.

| language     | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ------------ | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `css`        | 344 | 85.5% | 94.5% | 98.0%  | 0.903 | 0.922   | 1           | 2.0%      |
| `html`       | 84  | 25.0% | 33.3% | 47.6%  | 0.31  | 0.349   | 1           | 52.4%     |
| `javascript` | 358 | 74.9% | 88.3% | 92.5%  | 0.82  | 0.847   | 1           | 7.5%      |
| `markdown`   | 490 | 23.3% | 39.4% | 52.4%  | 0.327 | 0.374   | 2           | 47.6%     |
| `python`     | 504 | 62.9% | 82.1% | 93.1%  | 0.739 | 0.786   | 1           | 6.9%      |
| `rst`        | 98  | 19.4% | 22.4% | 45.9%  | 0.246 | 0.294   | 4           | 54.1%     |
| `rust`       | 138 | 54.3% | 74.6% | 87.0%  | 0.659 | 0.711   | 1           | 13.0%     |
| `sql`        | 92  | 51.1% | 69.6% | 93.5%  | 0.635 | 0.706   | 1           | 6.5%      |
| `typescript` | 249 | 58.6% | 85.9% | 94.4%  | 0.733 | 0.785   | 1           | 5.6%      |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 525 | 69.1% | 82.7% | 88.0%  | 0.764 | 0.793   | 1           | 12.0%     |
| `concept`   | 993 | 43.4% | 61.6% | 75.5%  | 0.542 | 0.594   | 1           | 24.5%     |
| `docstring` | 314 | 77.7% | 92.4% | 94.6%  | 0.85  | 0.875   | 1           | 5.4%      |
| `name`      | 525 | 50.1% | 65.3% | 78.3%  | 0.594 | 0.64    | 1           | 21.7%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n    | MRR   |
| --------- | ---- | ----- |
| full      | 9304 | 79.1% |
| truncated | 124  | 49.4% |

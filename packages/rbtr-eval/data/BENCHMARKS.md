# rbtr search-quality benchmark

Hit@k / MRR / NDCG@10 for queries against the rbtr index. See
`packages/rbtr-eval/README.md` for methodology.

## Run

Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.

| field         | value                               |
| ------------- | ----------------------------------- |
| seed          | 0                                   |
| sample target | 25 per (repo, language, provenance) |
| total queries | 2684                                |
| elapsed       | 46880 s                             |

## Repos

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 1616    | 125             |
| `astral-sh__uv`      | `cfe5277bc422` | 20163   | 350             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 13498   | 292             |
| `django__django`     | `e78a46a8fb29` | 60588   | 450             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 4811    | 310             |

## Headline metrics

No-expansion baseline (arm `none`) per repo. The expansion
ablation is in the section below.

| repo                 | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| -------------------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| **all repos**        | 2684 | 42.2% | 55.6% | 66.2%  | 0.501 | 0.54    | 1           | 33.8%     |
| `anthropics__skills` | 224  | 54.5% | 78.1% | 89.7%  | 0.672 | 0.727   | 1           | 10.3%     |
| `astral-sh__uv`      | 628  | 32.5% | 44.6% | 57.3%  | 0.401 | 0.442   | 1           | 42.7%     |
| `badlogic__pi-mono`  | 504  | 50.6% | 64.1% | 71.4%  | 0.581 | 0.613   | 1           | 28.6%     |
| `django__django`     | 809  | 36.2% | 46.2% | 56.6%  | 0.425 | 0.459   | 1           | 43.4%     |
| `rbtr__rbtr`         | 519  | 49.7% | 65.7% | 76.5%  | 0.589 | 0.632   | 1           | 23.5%     |

## Expansion ablation

MRR / Hit@k for each expansion arm (`none`, `keywords`,
`variants`, `both`) broken down by query kind, aggregated
across all repos. Compare arms within a kind to read the
effect of each channel.

| arm      | query kind   | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 |
| -------- | ------------ | ---- | ----- | ----- | ------ | ----- | ------- |
| both     | **all**      | 2684 | 41.4% | 56.5% | 66.3%  | 0.499 | 0.539   |
| keywords | **all**      | 2684 | 41.7% | 56.2% | 66.5%  | 0.502 | 0.541   |
| none     | **all**      | 2684 | 42.2% | 55.6% | 66.2%  | 0.501 | 0.54    |
| variants | **all**      | 2684 | 42.0% | 55.9% | 65.8%  | 0.5   | 0.539   |
| both     | `code`       | 414  | 70.0% | 85.0% | 90.8%  | 0.782 | 0.813   |
| keywords | `code`       | 414  | 70.0% | 84.5% | 90.6%  | 0.781 | 0.812   |
| none     | `code`       | 414  | 72.7% | 85.0% | 91.8%  | 0.799 | 0.829   |
| variants | `code`       | 414  | 72.5% | 85.3% | 91.8%  | 0.799 | 0.828   |
| both     | `concept`    | 1181 | 26.0% | 42.5% | 55.0%  | 0.355 | 0.402   |
| keywords | `concept`    | 1181 | 26.6% | 42.2% | 55.1%  | 0.359 | 0.406   |
| none     | `concept`    | 1181 | 24.9% | 39.2% | 51.0%  | 0.333 | 0.376   |
| variants | `concept`    | 1181 | 24.9% | 39.7% | 50.7%  | 0.334 | 0.376   |
| both     | `identifier` | 1089 | 47.2% | 60.8% | 69.2%  | 0.548 | 0.583   |
| keywords | `identifier` | 1089 | 47.4% | 60.7% | 69.7%  | 0.55  | 0.585   |
| none     | `identifier` | 1089 | 49.3% | 62.3% | 72.9%  | 0.57  | 0.608   |
| variants | `identifier` | 1089 | 49.0% | 62.4% | 72.4%  | 0.568 | 0.605   |

![MRR by repo](mrr_by_repo.png)

## Search latency

Shared index home size: **1.9 GiB**.

| repo                 | search P50 | search P95 |
| -------------------- | ---------- | ---------- |
| **all repos**        | 3394 ms    | 8721 ms    |
| `anthropics__skills` | 4653 ms    | 9948 ms    |
| `astral-sh__uv`      | 3232 ms    | 10852 ms   |
| `badlogic__pi-mono`  | 3364 ms    | 8018 ms    |
| `django__django`     | 3488 ms    | 7337 ms    |
| `rbtr__rbtr`         | 3062 ms    | 6981 ms    |

## Classification

Cross-tabulation of query provenance (where the query was sampled
from) against online `QueryKind` (how `classify_query` routes it
for expansion).

| provenance  | CONCEPT | IDENTIFIER | CODE  | n    |
| ----------- | ------- | ---------- | ----- | ---- |
| `body`      | 2.0%    | 39.8%      | 58.2% | 2600 |
| `concept`   | 99.6%   | 0.4%       | null  | 4628 |
| `docstring` | 1.8%    | 91.6%      | 6.6%  | 908  |
| `name`      | 1.8%    | 94.9%      | 3.2%  | 2600 |

## Per-language breakdown

Aggregated across all repos for each language present in the sample.

| language     | n   | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ------------ | --- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `css`        | 150 | 72.7% | 80.7% | 87.3%  | 0.777 | 0.801   | 1           | 12.7%     |
| `html`       | 95  | 29.5% | 40.0% | 50.5%  | 0.363 | 0.397   | 1           | 49.5%     |
| `javascript` | 241 | 69.7% | 83.0% | 86.7%  | 0.765 | 0.791   | 1           | 13.3%     |
| `json`       | 389 | 11.6% | 17.2% | 26.2%  | 0.156 | 0.181   | 2           | 73.8%     |
| `markdown`   | 494 | 21.5% | 37.9% | 50.2%  | 0.31  | 0.357   | 2           | 49.8%     |
| `python`     | 493 | 62.9% | 82.8% | 92.9%  | 0.736 | 0.784   | 1           | 7.1%      |
| `rst`        | 99  | 15.2% | 21.2% | 42.4%  | 0.213 | 0.262   | 3           | 57.6%     |
| `rust`       | 143 | 54.5% | 71.3% | 86.0%  | 0.651 | 0.701   | 1           | 14.0%     |
| `sql`        | 94  | 48.9% | 68.1% | 94.7%  | 0.625 | 0.701   | 1           | 5.3%      |
| `toml`       | 84  | 26.2% | 36.9% | 40.5%  | 0.31  | 0.333   | 1           | 59.5%     |
| `typescript` | 236 | 75.8% | 91.9% | 98.3%  | 0.843 | 0.878   | 1           | 1.7%      |
| `yaml`       | 166 | 15.7% | 22.3% | 36.1%  | 0.205 | 0.241   | 2           | 63.9%     |

## Per-provenance breakdown

Aggregated across all repos and languages for each provenance.

| provenance  | n    | Hit@1 | Hit@3 | Hit@10 | MRR   | NDCG@10 | median rank | not found |
| ----------- | ---- | ----- | ----- | ------ | ----- | ------- | ----------- | --------- |
| `body`      | 650  | 59.7% | 73.1% | 82.3%  | 0.676 | 0.712   | 1           | 17.7%     |
| `concept`   | 1157 | 25.1% | 39.8% | 51.1%  | 0.335 | 0.378   | 2           | 48.9%     |
| `docstring` | 227  | 78.0% | 91.2% | 93.8%  | 0.845 | 0.869   | 1           | 6.2%      |
| `name`      | 650  | 42.6% | 54.0% | 67.2%  | 0.5   | 0.541   | 1           | 32.8%     |

![MRR by provenance](mrr_by_provenance.png)

## Truncation impact

MRR breakdown by whether the target chunk's embedding was truncated
to fit the context window.

| embedding | n     | MRR   |
| --------- | ----- | ----- |
| full      | 10628 | 75.8% |
| truncated | 108   | 46.6% |

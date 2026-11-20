# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- **alpha** — semantic (embedding cosine similarity)
- **beta** — lexical (BM25 keyword search)
- **gamma** — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.071036 | 0.08094  | 0.848024 |
| identifier | 0.011332 | 0.204236 | 0.784432 |
| code       | 0.012328 | 0.813697 | 0.173975 |

## Result

One row per query kind. Each kind is tuned as its own
optimisation, against its own query set, so the rows are
three separate results rather than parts of one.

`std error` is the standard error of the delta, taken over
the per-query differences between the two arms. The verdict
is `recommend` only where the delta exceeds twice that
error; otherwise this run cannot tell the tuned weights
from the current ones, and the recommended config below
comments that kind out.

What the MRR measures: each trial reorders a pool of 50
candidates fetched once per query under the current default
weights, and scores the target's position in the top 10.
The reranker takes no part. `pool ceiling` is the share of
queries whose target reached that pool at all — a target the
defaults did not retrieve scores zero under every weight
triple, so the MRR cannot rise above the ceiling, and a
weight set whose merit would be retrieving something new
earns nothing for it here. These figures answer "which of
these weightings orders this pool best", not "how well does
search work", and are not comparable with the MRR in
`BENCHMARKS.md`.

| kind       | metric | current             | recommended         | delta                         | std error | verdict   | n   | pool ceiling |
| ---------- | ------ | ------------------- | ------------------- | ----------------------------- | --------- | --------- | --- | ------------ |
| concept    | MRR    | 0.15917290496984818 | 0.16084858251888817 | 0.0016756775490399944 (+1.1%) | ± 0.0045  | withhold  | 458 | 0.4607       |
| identifier | MRR    | 0.5193698922295831  | 0.530094558632014   | 0.010724666402430971 (+2.1%)  | ± 0.0049  | recommend | 841 | 0.8621       |
| code       | MRR    | 0.6449044928705945  | 0.6616182405165457  | 0.016713747645951238 (+2.6%)  | ± 0.0068  | recommend | 295 | 0.9017       |

## Top-trial spread

The range each weight covers across a study's ten
best-scoring trials, and the MRR those trials span. Wide
weight ranges beside a narrow `mrr_range` mean the
objective cannot separate the weights it is choosing
between, and the single triple above is one point drawn
from a plateau.

| kind       | n_top | alpha_min | alpha_max | beta_min | beta_max | gamma_min | gamma_max | mrr_range |
| ---------- | ----- | --------- | --------- | -------- | -------- | --------- | --------- | --------- |
| concept    | 10    | 0.064499  | 0.594595  | 0.08094  | 0.44287  | 0.009242  | 0.848024  | 0.002123  |
| identifier | 10    | 0.003532  | 0.024736  | 0.045981 | 0.47724  | 0.512216  | 0.947992  | 0.00526   |
| code       | 10    | 0.000149  | 0.035677  | 0.427222 | 0.976295 | 0.006065  | 0.566517  | 0.001751  |

## Impact by dimension

MRR and NDCG@10 under the current (baseline) and recommended
(best) weights: over every query, then per repo, language and
provenance.

| dimension  | value                | baseline MRR | best MRR | delta   | baseline NDCG@10 | best NDCG@10 | delta NDCG@10 |
| ---------- | -------------------- | ------------ | -------- | ------- | ---------------- | ------------ | ------------- |
| **all**    |                      | 0.4391       | 0.4483   | 0.0092  | 0.4806           | 0.4864       | 0.0058        |
| slug       | `anthropics__skills` | 0.5904       | 0.5977   | 0.0073  | 0.6386           | 0.6428       | 0.0042        |
| slug       | `astral-sh__uv`      | 0.4134       | 0.413    | -0.0005 | 0.449            | 0.4446       | -0.0044       |
| slug       | `badlogic__pi-mono`  | 0.4539       | 0.4646   | 0.0107  | 0.5011           | 0.5108       | 0.0098        |
| slug       | `django__django`     | 0.3335       | 0.3383   | 0.0048  | 0.3642           | 0.3662       | 0.0019        |
| slug       | `rbtr__rbtr`         | 0.429        | 0.4624   | 0.0334  | 0.4819           | 0.5083       | 0.0265        |
| language   | `bash`               | 0.5745       | 0.5826   | 0.0081  | 0.6185           | 0.6255       | 0.007         |
| language   | `css`                | 0.6394       | 0.6302   | -0.0092 | 0.6841           | 0.6732       | -0.0109       |
| language   | `html`               | 0.1998       | 0.2504   | 0.0506  | 0.2329           | 0.2791       | 0.0462        |
| language   | `javascript`         | 0.5237       | 0.5179   | -0.0057 | 0.564            | 0.5554       | -0.0086       |
| language   | `json`               | 0.1876       | 0.1927   | 0.0051  | 0.2205           | 0.2242       | 0.0037        |
| language   | `markdown`           | 0.2127       | 0.2756   | 0.0629  | 0.2463           | 0.3071       | 0.0609        |
| language   | `plaintext`          | 0.1891       | 0.2038   | 0.0147  | 0.2199           | 0.2298       | 0.0099        |
| language   | `python`             | 0.605        | 0.6105   | 0.0055  | 0.6515           | 0.6515       | -0.0          |
| language   | `rst`                | 0.1094       | 0.1117   | 0.0023  | 0.1219           | 0.1277       | 0.0057        |
| language   | `rust`               | 0.5773       | 0.576    | -0.0013 | 0.6332           | 0.62         | -0.0131       |
| language   | `sql`                | 0.5502       | 0.5929   | 0.0427  | 0.6229           | 0.65         | 0.0271        |
| language   | `toml`               | 0.1193       | 0.101    | -0.0182 | 0.1465           | 0.123        | -0.0235       |
| language   | `typescript`         | 0.5884       | 0.5925   | 0.0041  | 0.6437           | 0.6414       | -0.0023       |
| language   | `yaml`               | 0.1193       | 0.1146   | -0.0047 | 0.1494           | 0.1455       | -0.0039       |
| provenance | `body`               | 0.5104       | 0.5399   | 0.0294  | 0.5579           | 0.581        | 0.0231        |
| provenance | `concept`            | 0.1371       | 0.1307   | -0.0063 | 0.1748           | 0.1651       | -0.0097       |
| provenance | `docstring`          | 0.6792       | 0.6674   | -0.0118 | 0.7166           | 0.7038       | -0.0128       |
| provenance | `name`               | 0.5076       | 0.515    | 0.0074  | 0.5462           | 0.5532       | 0.007         |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.identifier]
alpha = 0.011331888722881895
beta = 0.20423561240938556
gamma = 0.7844324988677325

[search_weights.code]
alpha = 0.012328326021083746
beta = 0.8136969141958394
gamma = 0.17397475978307678
# [search_weights.concept]
# alpha = 0.07103605819788694
# beta = 0.08093997769720093
# gamma = 0.8480239641049121
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 1594
- elapsed: 16880 s

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills | bash       | body       | 16        |
| anthropics__skills | bash       | concept    | 10        |
| anthropics__skills | bash       | docstring  | 6         |
| anthropics__skills | bash       | name       | 10        |
| anthropics__skills | css        | body       | 11        |
| anthropics__skills | css        | concept    | 11        |
| anthropics__skills | css        | docstring  | 10        |
| anthropics__skills | css        | name       | 11        |
| anthropics__skills | javascript | body       | 18        |
| anthropics__skills | javascript | concept    | 10        |
| anthropics__skills | javascript | docstring  | 10        |
| anthropics__skills | javascript | name       | 10        |
| anthropics__skills | json       | body       | 10        |
| anthropics__skills | json       | concept    | 11        |
| anthropics__skills | json       | name       | 10        |
| anthropics__skills | markdown   | body       | 10        |
| anthropics__skills | markdown   | concept    | 10        |
| anthropics__skills | markdown   | name       | 10        |
| anthropics__skills | plaintext  | body       | 10        |
| anthropics__skills | plaintext  | concept    | 10        |
| anthropics__skills | python     | body       | 22        |
| anthropics__skills | python     | concept    | 10        |
| anthropics__skills | python     | docstring  | 13        |
| anthropics__skills | python     | name       | 10        |
| anthropics__skills | typescript | body       | 20        |
| anthropics__skills | typescript | concept    | 10        |
| anthropics__skills | typescript | docstring  | 9         |
| anthropics__skills | typescript | name       | 10        |
| astral-sh__uv      | bash       | body       | 23        |
| astral-sh__uv      | bash       | concept    | 10        |
| astral-sh__uv      | bash       | docstring  | 10        |
| astral-sh__uv      | bash       | name       | 10        |
| astral-sh__uv      | css        | body       | 11        |
| astral-sh__uv      | css        | concept    | 10        |
| astral-sh__uv      | css        | docstring  | 10        |
| astral-sh__uv      | css        | name       | 12        |
| astral-sh__uv      | json       | body       | 10        |
| astral-sh__uv      | json       | concept    | 10        |
| astral-sh__uv      | json       | name       | 10        |
| astral-sh__uv      | markdown   | body       | 10        |
| astral-sh__uv      | markdown   | concept    | 10        |
| astral-sh__uv      | markdown   | name       | 10        |
| astral-sh__uv      | plaintext  | body       | 10        |
| astral-sh__uv      | plaintext  | concept    | 10        |
| astral-sh__uv      | python     | body       | 20        |
| astral-sh__uv      | python     | concept    | 10        |
| astral-sh__uv      | python     | docstring  | 14        |
| astral-sh__uv      | python     | name       | 10        |
| astral-sh__uv      | rst        | body       | 10        |
| astral-sh__uv      | rst        | concept    | 10        |
| astral-sh__uv      | rst        | name       | 5         |
| astral-sh__uv      | rust       | body       | 20        |
| astral-sh__uv      | rust       | concept    | 10        |
| astral-sh__uv      | rust       | docstring  | 17        |
| astral-sh__uv      | rust       | name       | 10        |
| astral-sh__uv      | toml       | body       | 16        |
| astral-sh__uv      | toml       | concept    | 10        |
| astral-sh__uv      | toml       | docstring  | 10        |
| astral-sh__uv      | toml       | name       | 10        |
| astral-sh__uv      | yaml       | body       | 11        |
| astral-sh__uv      | yaml       | concept    | 10        |
| astral-sh__uv      | yaml       | docstring  | 10        |
| astral-sh__uv      | yaml       | name       | 10        |
| badlogic__pi-mono  | bash       | body       | 24        |
| badlogic__pi-mono  | bash       | concept    | 10        |
| badlogic__pi-mono  | bash       | docstring  | 14        |
| badlogic__pi-mono  | bash       | name       | 10        |
| badlogic__pi-mono  | css        | body       | 13        |
| badlogic__pi-mono  | css        | concept    | 10        |
| badlogic__pi-mono  | css        | docstring  | 10        |
| badlogic__pi-mono  | css        | name       | 10        |
| badlogic__pi-mono  | javascript | body       | 21        |
| badlogic__pi-mono  | javascript | concept    | 10        |
| badlogic__pi-mono  | javascript | docstring  | 10        |
| badlogic__pi-mono  | javascript | name       | 10        |
| badlogic__pi-mono  | json       | body       | 10        |
| badlogic__pi-mono  | json       | concept    | 10        |
| badlogic__pi-mono  | json       | name       | 10        |
| badlogic__pi-mono  | markdown   | body       | 10        |
| badlogic__pi-mono  | markdown   | concept    | 10        |
| badlogic__pi-mono  | markdown   | name       | 10        |
| badlogic__pi-mono  | typescript | body       | 20        |
| badlogic__pi-mono  | typescript | concept    | 11        |
| badlogic__pi-mono  | typescript | docstring  | 13        |
| badlogic__pi-mono  | typescript | name       | 10        |
| django__django     | bash       | body       | 18        |
| django__django     | bash       | concept    | 10        |
| django__django     | bash       | name       | 10        |
| django__django     | css        | body       | 20        |
| django__django     | css        | concept    | 11        |
| django__django     | css        | docstring  | 10        |
| django__django     | css        | name       | 12        |
| django__django     | html       | body       | 12        |
| django__django     | html       | concept    | 10        |
| django__django     | html       | name       | 10        |
| django__django     | javascript | body       | 21        |
| django__django     | javascript | concept    | 10        |
| django__django     | javascript | docstring  | 11        |
| django__django     | javascript | name       | 10        |
| django__django     | json       | body       | 10        |
| django__django     | json       | concept    | 10        |
| django__django     | json       | name       | 10        |
| django__django     | markdown   | body       | 10        |
| django__django     | markdown   | concept    | 10        |
| django__django     | markdown   | name       | 10        |
| django__django     | plaintext  | body       | 10        |
| django__django     | plaintext  | concept    | 10        |
| django__django     | python     | body       | 22        |
| django__django     | python     | concept    | 10        |
| django__django     | python     | docstring  | 15        |
| django__django     | python     | name       | 10        |
| django__django     | rst        | body       | 10        |
| django__django     | rst        | concept    | 10        |
| django__django     | rst        | name       | 10        |
| django__django     | yaml       | body       | 10        |
| django__django     | yaml       | concept    | 10        |
| django__django     | yaml       | docstring  | 1         |
| django__django     | yaml       | name       | 10        |
| rbtr__rbtr         | bash       | body       | 15        |
| rbtr__rbtr         | bash       | concept    | 10        |
| rbtr__rbtr         | json       | body       | 10        |
| rbtr__rbtr         | json       | concept    | 10        |
| rbtr__rbtr         | json       | name       | 10        |
| rbtr__rbtr         | markdown   | body       | 10        |
| rbtr__rbtr         | markdown   | concept    | 12        |
| rbtr__rbtr         | markdown   | name       | 10        |
| rbtr__rbtr         | python     | body       | 21        |
| rbtr__rbtr         | python     | concept    | 11        |
| rbtr__rbtr         | python     | docstring  | 11        |
| rbtr__rbtr         | python     | name       | 10        |
| rbtr__rbtr         | sql        | body       | 16        |
| rbtr__rbtr         | sql        | concept    | 10        |
| rbtr__rbtr         | sql        | docstring  | 10        |
| rbtr__rbtr         | sql        | name       | 10        |
| rbtr__rbtr         | typescript | body       | 20        |
| rbtr__rbtr         | typescript | concept    | 11        |
| rbtr__rbtr         | typescript | docstring  | 11        |
| rbtr__rbtr         | typescript | name       | 10        |

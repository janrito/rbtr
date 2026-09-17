# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.174711 | 0.337455 | 0.487834 |
| identifier | 0.046651 | 0.358908 | 0.594441 |
| code       | 0.003365 | 0.768612 | 0.228023 |

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

| kind       | metric | current            | recommended         | delta                        | std error | verdict  | n   | pool ceiling |
| ---------- | ------ | ------------------ | ------------------- | ---------------------------- | --------- | -------- | --- | ------------ |
| concept    | MRR    | 0.1633083801206072 | 0.17238424481874265 | 0.009075864698135455 (+5.6%) | ± 0.0067  | withhold | 458 | 0.4694       |
| identifier | MRR    | 0.516808658692545  | 0.5201196118257729  | 0.003310953133227934 (+0.6%) | ± 0.0036  | withhold | 844 | 0.8507       |
| code       | MRR    | 0.6529042238364272 | 0.6571643798762443  | 0.004260156039817109 (+0.7%) | ± 0.0069  | withhold | 295 | 0.9051       |

## Top-trial spread

The range each weight covers across a study's ten
best-scoring trials, and the MRR those trials span. Wide
weight ranges beside a narrow `mrr_range` mean the
objective cannot separate the weights it is choosing
between, and the single triple above is one point drawn
from a plateau.

| kind       | n_top | alpha_min | alpha_max | beta_min | beta_max | gamma_min | gamma_max | mrr_range |
| ---------- | ----- | --------- | --------- | -------- | -------- | --------- | --------- | --------- |
| concept    | 10    | 0.020334  | 0.288018  | 0.033426 | 0.545142 | 0.189889  | 0.946239  | 0.002644  |
| identifier | 10    | 0.000598  | 0.083052  | 0.291555 | 0.417622 | 0.499326  | 0.67984   | 0.003137  |
| code       | 10    | 0.00008   | 0.073596  | 0.558439 | 0.816695 | 0.176449  | 0.439903  | 0.002194  |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            | __all__    | __all__    | 0.440569     | 0.445708 | 0.0051  | 0.47827             | 0.481559        | 0.0033           |
| __all__            | __all__    | body       | 0.514204     | 0.528009 | 0.0138  | 0.557769            | 0.569791        | 0.012            |
| __all__            | __all__    | concept    | 0.138956     | 0.134001 | -0.005  | 0.168821            | 0.161729        | -0.0071          |
| __all__            | __all__    | docstring  | 0.681239     | 0.687026 | 0.0058  | 0.716229            | 0.718752        | 0.0025           |
| __all__            | __all__    | name       | 0.506922     | 0.508937 | 0.002   | 0.545713            | 0.547015        | 0.0013           |
| __all__            | bash       | __all__    | 0.58529      | 0.586351 | 0.0011  | 0.624956            | 0.628316        | 0.0034           |
| __all__            | css        | __all__    | 0.662346     | 0.660794 | -0.0016 | 0.701892            | 0.697979        | -0.0039          |
| __all__            | html       | __all__    | 0.205593     | 0.260714 | 0.0551  | 0.237454            | 0.287729        | 0.0503           |
| __all__            | javascript | __all__    | 0.540579     | 0.522764 | -0.0178 | 0.581582            | 0.554473        | -0.0271          |
| __all__            | json       | __all__    | 0.183572     | 0.173912 | -0.0097 | 0.214087            | 0.2067          | -0.0074          |
| __all__            | markdown   | __all__    | 0.19144      | 0.234524 | 0.0431  | 0.221742            | 0.26742         | 0.0457           |
| __all__            | plaintext  | __all__    | 0.185278     | 0.19213  | 0.0069  | 0.205292            | 0.225115        | 0.0198           |
| __all__            | python     | __all__    | 0.614008     | 0.621537 | 0.0075  | 0.653926            | 0.655647        | 0.0017           |
| __all__            | rst        | __all__    | 0.094618     | 0.098557 | 0.0039  | 0.110597            | 0.113914        | 0.0033           |
| __all__            | rust       | __all__    | 0.532331     | 0.538158 | 0.0058  | 0.582221            | 0.582633        | 0.0004           |
| __all__            | sql        | __all__    | 0.567633     | 0.566304 | -0.0013 | 0.6366              | 0.635387        | -0.0012          |
| __all__            | toml       | __all__    | 0.117478     | 0.109179 | -0.0083 | 0.140645            | 0.139156        | -0.0015          |
| __all__            | typescript | __all__    | 0.580708     | 0.586433 | 0.0057  | 0.63318             | 0.630696        | -0.0025          |
| __all__            | yaml       | __all__    | 0.123313     | 0.13534  | 0.012   | 0.149886            | 0.155916        | 0.006            |
| anthropics__skills | __all__    | __all__    | 0.587818     | 0.591467 | 0.0036  | 0.634233            | 0.63522         | 0.001            |
| astral-sh__uv      | __all__    | __all__    | 0.417675     | 0.423766 | 0.0061  | 0.44844             | 0.452633        | 0.0042           |
| badlogic__pi-mono  | __all__    | __all__    | 0.451605     | 0.452233 | 0.0006  | 0.492064            | 0.491911        | -0.0002          |
| django__django     | __all__    | __all__    | 0.336528     | 0.333213 | -0.0033 | 0.364651            | 0.362274        | -0.0024          |
| rbtr__rbtr         | __all__    | __all__    | 0.433259     | 0.457335 | 0.0241  | 0.483013            | 0.500769        | 0.0178           |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
# [search_weights.concept]
# alpha = 0.17471104556180572
# beta = 0.33745479350707447
# gamma = 0.48783416093111986
#
# [search_weights.identifier]
# alpha = 0.04665051374126059
# beta = 0.35890803521278397
# gamma = 0.5944414510459555
#
# [search_weights.code]
# alpha = 0.0033654757256979563
# beta = 0.7686116381960251
# gamma = 0.2280228860782769
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 1597
- elapsed: 16112 s

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills | bash       | body       | 16        |
| anthropics__skills | bash       | concept    | 10        |
| anthropics__skills | bash       | docstring  | 6         |
| anthropics__skills | bash       | name       | 10        |
| anthropics__skills | css        | body       | 11        |
| anthropics__skills | css        | concept    | 10        |
| anthropics__skills | css        | docstring  | 10        |
| anthropics__skills | css        | name       | 11        |
| anthropics__skills | javascript | body       | 18        |
| anthropics__skills | javascript | concept    | 10        |
| anthropics__skills | javascript | docstring  | 10        |
| anthropics__skills | javascript | name       | 10        |
| anthropics__skills | json       | body       | 10        |
| anthropics__skills | json       | concept    | 10        |
| anthropics__skills | json       | name       | 10        |
| anthropics__skills | markdown   | body       | 10        |
| anthropics__skills | markdown   | concept    | 11        |
| anthropics__skills | markdown   | name       | 10        |
| anthropics__skills | plaintext  | body       | 10        |
| anthropics__skills | plaintext  | concept    | 10        |
| anthropics__skills | python     | body       | 22        |
| anthropics__skills | python     | concept    | 11        |
| anthropics__skills | python     | docstring  | 13        |
| anthropics__skills | python     | name       | 10        |
| anthropics__skills | typescript | body       | 20        |
| anthropics__skills | typescript | concept    | 10        |
| anthropics__skills | typescript | docstring  | 9         |
| anthropics__skills | typescript | name       | 10        |
| astral-sh__uv      | bash       | body       | 23        |
| astral-sh__uv      | bash       | concept    | 11        |
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
| badlogic__pi-mono  | markdown   | concept    | 14        |
| badlogic__pi-mono  | markdown   | name       | 10        |
| badlogic__pi-mono  | typescript | body       | 20        |
| badlogic__pi-mono  | typescript | concept    | 10        |
| badlogic__pi-mono  | typescript | docstring  | 13        |
| badlogic__pi-mono  | typescript | name       | 10        |
| django__django     | bash       | body       | 18        |
| django__django     | bash       | concept    | 10        |
| django__django     | bash       | name       | 10        |
| django__django     | css        | body       | 20        |
| django__django     | css        | concept    | 10        |
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
| django__django     | json       | concept    | 11        |
| django__django     | json       | name       | 10        |
| django__django     | markdown   | body       | 10        |
| django__django     | markdown   | concept    | 10        |
| django__django     | markdown   | name       | 10        |
| django__django     | plaintext  | body       | 10        |
| django__django     | plaintext  | concept    | 10        |
| django__django     | python     | body       | 22        |
| django__django     | python     | concept    | 11        |
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
| rbtr__rbtr         | markdown   | concept    | 11        |
| rbtr__rbtr         | markdown   | name       | 10        |
| rbtr__rbtr         | python     | body       | 21        |
| rbtr__rbtr         | python     | concept    | 10        |
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

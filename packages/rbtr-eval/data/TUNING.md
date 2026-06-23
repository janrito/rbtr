# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.158571 | 0.624253 | 0.217176 |
| identifier | 0.000435 | 0.297442 | 0.702123 |
| code       | 0.000359 | 0.371287 | 0.628354 |

## Result

| metric | current            | recommended        | delta                        |
| ------ | ------------------ | ------------------ | ---------------------------- |
| MRR    | 0.2813279101232363 | 0.2831475585341928 | 0.001819648410956498 (+0.6%) |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            | __all__    | __all__    | 0.402624     | 0.416546 | 0.0139  | 0.437178            | 0.450506        | 0.0133           |
| __all__            | __all__    | body       | 0.491702     | 0.519484 | 0.0278  | 0.537815            | 0.564931        | 0.0271           |
| __all__            | __all__    | concept    | 0.176559     | 0.17566  | -0.0009 | 0.196819            | 0.198549        | 0.0017           |
| __all__            | __all__    | docstring  | 0.691896     | 0.729581 | 0.0377  | 0.737244            | 0.764616        | 0.0274           |
| __all__            | __all__    | name       | 0.435473     | 0.4418   | 0.0063  | 0.468875            | 0.474959        | 0.0061           |
| __all__            | css        | __all__    | 0.598611     | 0.663889 | 0.0653  | 0.615907            | 0.68041         | 0.0645           |
| __all__            | html       | __all__    | 0.343056     | 0.381481 | 0.0384  | 0.36488             | 0.402096        | 0.0372           |
| __all__            | javascript | __all__    | 0.593125     | 0.553175 | -0.04   | 0.617062            | 0.580088        | -0.037           |
| __all__            | json       | __all__    | 0.147163     | 0.162245 | 0.0151  | 0.163946            | 0.181418        | 0.0175           |
| __all__            | markdown   | __all__    | 0.187968     | 0.214349 | 0.0264  | 0.227176            | 0.251233        | 0.0241           |
| __all__            | python     | __all__    | 0.585551     | 0.590558 | 0.005   | 0.640253            | 0.641741        | 0.0015           |
| __all__            | rst        | __all__    | 0.044762     | 0.054762 | 0.01    | 0.05734             | 0.065475        | 0.0081           |
| __all__            | rust       | __all__    | 0.645833     | 0.65506  | 0.0092  | 0.696612            | 0.695837        | -0.0008          |
| __all__            | toml       | __all__    | 0.21037      | 0.250926 | 0.0406  | 0.255405            | 0.287054        | 0.0316           |
| __all__            | typescript | __all__    | 0.782738     | 0.788889 | 0.0062  | 0.822461            | 0.833519        | 0.0111           |
| __all__            | yaml       | __all__    | 0.148274     | 0.162857 | 0.0146  | 0.178823            | 0.198436        | 0.0196           |
| anthropics__skills | __all__    | __all__    | 0.532018     | 0.538764 | 0.0067  | 0.578865            | 0.591343        | 0.0125           |
| astral-sh__uv      | __all__    | __all__    | 0.367901     | 0.376724 | 0.0088  | 0.399851            | 0.407127        | 0.0073           |
| badlogic__pi-mono  | __all__    | __all__    | 0.514573     | 0.544043 | 0.0295  | 0.542363            | 0.567219        | 0.0249           |
| django__django     | __all__    | __all__    | 0.284002     | 0.289803 | 0.0058  | 0.318134            | 0.324354        | 0.0062           |
| rbtr__rbtr         | __all__    | __all__    | 0.471891     | 0.492885 | 0.021   | 0.513014            | 0.534618        | 0.0216           |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.concept]
alpha = 0.1585710308538254
beta = 0.6242534503569727
gamma = 0.21717551878920197

[search_weights.identifier]
alpha = 0.00043521330346418807
beta = 0.29744208652656373
gamma = 0.7021227001699721

[search_weights.code]
alpha = 0.0003590684753362696
beta = 0.3712866982407882
gamma = 0.6283542332838755
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 840
- elapsed: 5357 s

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills | markdown   | body       | 10        |
| anthropics__skills | markdown   | concept    | 10        |
| anthropics__skills | markdown   | name       | 10        |
| anthropics__skills | python     | body       | 10        |
| anthropics__skills | python     | concept    | 10        |
| anthropics__skills | python     | docstring  | 10        |
| anthropics__skills | python     | name       | 10        |
| astral-sh__uv      | json       | body       | 10        |
| astral-sh__uv      | json       | concept    | 10        |
| astral-sh__uv      | json       | name       | 10        |
| astral-sh__uv      | markdown   | body       | 10        |
| astral-sh__uv      | markdown   | concept    | 10        |
| astral-sh__uv      | markdown   | name       | 10        |
| astral-sh__uv      | python     | body       | 10        |
| astral-sh__uv      | python     | concept    | 10        |
| astral-sh__uv      | python     | docstring  | 10        |
| astral-sh__uv      | python     | name       | 10        |
| astral-sh__uv      | rust       | body       | 10        |
| astral-sh__uv      | rust       | concept    | 10        |
| astral-sh__uv      | rust       | docstring  | 10        |
| astral-sh__uv      | rust       | name       | 10        |
| astral-sh__uv      | toml       | body       | 10        |
| astral-sh__uv      | toml       | concept    | 10        |
| astral-sh__uv      | toml       | name       | 10        |
| astral-sh__uv      | yaml       | body       | 10        |
| astral-sh__uv      | yaml       | concept    | 10        |
| astral-sh__uv      | yaml       | name       | 10        |
| badlogic__pi-mono  | css        | body       | 10        |
| badlogic__pi-mono  | css        | concept    | 10        |
| badlogic__pi-mono  | css        | name       | 10        |
| badlogic__pi-mono  | javascript | body       | 10        |
| badlogic__pi-mono  | javascript | concept    | 10        |
| badlogic__pi-mono  | javascript | docstring  | 10        |
| badlogic__pi-mono  | javascript | name       | 10        |
| badlogic__pi-mono  | json       | body       | 10        |
| badlogic__pi-mono  | json       | concept    | 10        |
| badlogic__pi-mono  | json       | name       | 10        |
| badlogic__pi-mono  | markdown   | body       | 10        |
| badlogic__pi-mono  | markdown   | concept    | 10        |
| badlogic__pi-mono  | markdown   | name       | 10        |
| badlogic__pi-mono  | typescript | body       | 10        |
| badlogic__pi-mono  | typescript | concept    | 10        |
| badlogic__pi-mono  | typescript | docstring  | 10        |
| badlogic__pi-mono  | typescript | name       | 10        |
| django__django     | css        | body       | 10        |
| django__django     | css        | concept    | 10        |
| django__django     | css        | name       | 10        |
| django__django     | html       | body       | 10        |
| django__django     | html       | concept    | 10        |
| django__django     | html       | name       | 10        |
| django__django     | javascript | body       | 10        |
| django__django     | javascript | concept    | 10        |
| django__django     | javascript | docstring  | 10        |
| django__django     | javascript | name       | 10        |
| django__django     | json       | body       | 10        |
| django__django     | json       | concept    | 10        |
| django__django     | json       | name       | 10        |
| django__django     | markdown   | body       | 10        |
| django__django     | markdown   | concept    | 10        |
| django__django     | markdown   | name       | 10        |
| django__django     | python     | body       | 10        |
| django__django     | python     | concept    | 10        |
| django__django     | python     | docstring  | 10        |
| django__django     | python     | name       | 10        |
| django__django     | rst        | body       | 10        |
| django__django     | rst        | concept    | 10        |
| django__django     | rst        | name       | 10        |
| django__django     | yaml       | body       | 10        |
| django__django     | yaml       | concept    | 10        |
| django__django     | yaml       | name       | 10        |
| rbtr__rbtr         | json       | body       | 10        |
| rbtr__rbtr         | json       | concept    | 10        |
| rbtr__rbtr         | json       | name       | 10        |
| rbtr__rbtr         | markdown   | body       | 10        |
| rbtr__rbtr         | markdown   | concept    | 10        |
| rbtr__rbtr         | markdown   | name       | 10        |
| rbtr__rbtr         | python     | body       | 10        |
| rbtr__rbtr         | python     | concept    | 10        |
| rbtr__rbtr         | python     | docstring  | 10        |
| rbtr__rbtr         | python     | name       | 10        |
| rbtr__rbtr         | typescript | body       | 10        |
| rbtr__rbtr         | typescript | concept    | 10        |
| rbtr__rbtr         | typescript | docstring  | 10        |
| rbtr__rbtr         | typescript | name       | 10        |

# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.137519 | 0.27806  | 0.584421 |
| identifier | 0.007601 | 0.092751 | 0.899648 |
| code       | 0.059404 | 0.846227 | 0.09437  |

## Result

| metric | current             | recommended         | delta                        |
| ------ | ------------------- | ------------------- | ---------------------------- |
| MRR    | 0.19610506250229112 | 0.20816103962755234 | 0.012055977125261214 (+6.1%) |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            |            | __all__    | 0.443817     | 0.421326 | -0.0225 | 0.471481            | 0.448494        | -0.023           |
| __all__            | __all__    | __all__    | 0.463241     | 0.469329 | 0.0061  | 0.499421            | 0.504203        | 0.0048           |
| __all__            | __all__    | body       | 0.496538     | 0.511273 | 0.0147  | 0.540654            | 0.553461        | 0.0128           |
| __all__            | __all__    | concept    | 0.177647     | 0.172075 | -0.0056 | 0.210482            | 0.203629        | -0.0069          |
| __all__            | __all__    | docstring  | 0.695852     | 0.692136 | -0.0037 | 0.727141            | 0.722689        | -0.0045          |
| __all__            | __all__    | name       | 0.569728     | 0.581262 | 0.0115  | 0.600786            | 0.611546        | 0.0108           |
| __all__            | bash       | __all__    | 0.615487     | 0.646278 | 0.0308  | 0.668275            | 0.697151        | 0.0289           |
| __all__            | css        | __all__    | 0.697159     | 0.682232 | -0.0149 | 0.730645            | 0.718068        | -0.0126          |
| __all__            | html       | __all__    | 0.3234375    | 0.34375  | 0.0203  | 0.343175            | 0.358973        | 0.0158           |
| __all__            | javascript | __all__    | 0.570713     | 0.569053 | -0.0017 | 0.606333            | 0.603462        | -0.0029          |
| __all__            | json       | __all__    | 0.180471     | 0.178397 | -0.0021 | 0.210442            | 0.207704        | -0.0027          |
| __all__            | markdown   | __all__    | 0.205274     | 0.239166 | 0.0339  | 0.225527            | 0.26164         | 0.0361           |
| __all__            | python     | __all__    | 0.598179     | 0.615969 | 0.0178  | 0.641475            | 0.651835        | 0.0104           |
| __all__            | rst        | __all__    | 0.004608     | 0.009985 | 0.0054  | 0.010753            | 0.022243        | 0.0115           |
| __all__            | rust       | __all__    | 0.581184     | 0.569991 | -0.0112 | 0.640342            | 0.627213        | -0.0131          |
| __all__            | sql        | __all__    | 0.62284      | 0.651111 | 0.0283  | 0.68853             | 0.709852        | 0.0213           |
| __all__            | toml       | __all__    | 0.167907     | 0.138393 | -0.0295 | 0.191943            | 0.156           | -0.0359          |
| __all__            | typescript | __all__    | 0.582346     | 0.57432  | -0.008  | 0.625416            | 0.617708        | -0.0077          |
| __all__            | yaml       | __all__    | 0.14569      | 0.169599 | 0.0239  | 0.173153            | 0.1981          | 0.0249           |
| anthropics__skills | __all__    | __all__    | 0.579295     | 0.578914 | -0.0004 | 0.629008            | 0.625995        | -0.003           |
| astral-sh__uv      | __all__    | __all__    | 0.453504     | 0.459425 | 0.0059  | 0.484416            | 0.489217        | 0.0048           |
| badlogic__pi-mono  | __all__    | __all__    | 0.484073     | 0.486139 | 0.0021  | 0.521001            | 0.52283         | 0.0018           |
| django__django     | __all__    | __all__    | 0.346841     | 0.352476 | 0.0056  | 0.368274            | 0.373715        | 0.0054           |
| rbtr__rbtr         | __all__    | __all__    | 0.47081      | 0.493006 | 0.0222  | 0.519481            | 0.538844        | 0.0194           |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.concept]
alpha = 0.13751900829806735
beta = 0.27805965902468166
gamma = 0.584421332677251

[search_weights.identifier]
alpha = 0.0076011360583828486
beta = 0.09275092124464959
gamma = 0.8996479426969676

[search_weights.code]
alpha = 0.059403827118493804
beta = 0.8462265911073973
gamma = 0.09436958177410891
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 1554
- elapsed: 8606 s

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills |            | body       | 10        |
| anthropics__skills |            | concept    | 10        |
| anthropics__skills |            | name       | 10        |
| anthropics__skills | bash       | body       | 15        |
| anthropics__skills | bash       | concept    | 10        |
| anthropics__skills | bash       | docstring  | 6         |
| anthropics__skills | bash       | name       | 10        |
| anthropics__skills | css        | body       | 11        |
| anthropics__skills | css        | concept    | 10        |
| anthropics__skills | css        | docstring  | 10        |
| anthropics__skills | css        | name       | 11        |
| anthropics__skills | javascript | body       | 18        |
| anthropics__skills | javascript | concept    | 12        |
| anthropics__skills | javascript | docstring  | 10        |
| anthropics__skills | javascript | name       | 10        |
| anthropics__skills | json       | body       | 10        |
| anthropics__skills | json       | concept    | 10        |
| anthropics__skills | json       | name       | 10        |
| anthropics__skills | markdown   | body       | 10        |
| anthropics__skills | markdown   | concept    | 10        |
| anthropics__skills | markdown   | name       | 10        |
| anthropics__skills | python     | body       | 20        |
| anthropics__skills | python     | concept    | 12        |
| anthropics__skills | python     | docstring  | 13        |
| anthropics__skills | python     | name       | 10        |
| anthropics__skills | typescript | body       | 20        |
| anthropics__skills | typescript | concept    | 10        |
| anthropics__skills | typescript | docstring  | 9         |
| anthropics__skills | typescript | name       | 10        |
| astral-sh__uv      |            | body       | 10        |
| astral-sh__uv      |            | concept    | 11        |
| astral-sh__uv      |            | name       | 10        |
| astral-sh__uv      | bash       | body       | 21        |
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
| astral-sh__uv      | markdown   | body       | 16        |
| astral-sh__uv      | markdown   | concept    | 10        |
| astral-sh__uv      | markdown   | name       | 10        |
| astral-sh__uv      | python     | body       | 20        |
| astral-sh__uv      | python     | concept    | 10        |
| astral-sh__uv      | python     | docstring  | 14        |
| astral-sh__uv      | python     | name       | 10        |
| astral-sh__uv      | rust       | body       | 20        |
| astral-sh__uv      | rust       | concept    | 11        |
| astral-sh__uv      | rust       | docstring  | 12        |
| astral-sh__uv      | rust       | name       | 10        |
| astral-sh__uv      | toml       | body       | 17        |
| astral-sh__uv      | toml       | concept    | 11        |
| astral-sh__uv      | toml       | docstring  | 10        |
| astral-sh__uv      | toml       | name       | 10        |
| astral-sh__uv      | yaml       | body       | 11        |
| astral-sh__uv      | yaml       | concept    | 10        |
| astral-sh__uv      | yaml       | docstring  | 10        |
| astral-sh__uv      | yaml       | name       | 10        |
| badlogic__pi-mono  | bash       | body       | 22        |
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
| badlogic__pi-mono  | markdown   | body       | 12        |
| badlogic__pi-mono  | markdown   | concept    | 14        |
| badlogic__pi-mono  | markdown   | name       | 11        |
| badlogic__pi-mono  | typescript | body       | 20        |
| badlogic__pi-mono  | typescript | concept    | 11        |
| badlogic__pi-mono  | typescript | docstring  | 13        |
| badlogic__pi-mono  | typescript | name       | 10        |
| django__django     |            | body       | 10        |
| django__django     |            | concept    | 12        |
| django__django     |            | name       | 10        |
| django__django     | css        | body       | 20        |
| django__django     | css        | concept    | 12        |
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
| django__django     | markdown   | body       | 12        |
| django__django     | markdown   | concept    | 10        |
| django__django     | markdown   | name       | 12        |
| django__django     | python     | body       | 22        |
| django__django     | python     | concept    | 13        |
| django__django     | python     | docstring  | 15        |
| django__django     | python     | name       | 10        |
| django__django     | rst        | body       | 10        |
| django__django     | rst        | concept    | 11        |
| django__django     | rst        | name       | 10        |
| django__django     | yaml       | body       | 10        |
| django__django     | yaml       | concept    | 10        |
| django__django     | yaml       | docstring  | 1         |
| django__django     | yaml       | name       | 10        |
| rbtr__rbtr         | json       | body       | 10        |
| rbtr__rbtr         | json       | concept    | 10        |
| rbtr__rbtr         | json       | name       | 10        |
| rbtr__rbtr         | markdown   | body       | 11        |
| rbtr__rbtr         | markdown   | concept    | 10        |
| rbtr__rbtr         | markdown   | name       | 11        |
| rbtr__rbtr         | python     | body       | 21        |
| rbtr__rbtr         | python     | concept    | 13        |
| rbtr__rbtr         | python     | docstring  | 11        |
| rbtr__rbtr         | python     | name       | 10        |
| rbtr__rbtr         | sql        | body       | 15        |
| rbtr__rbtr         | sql        | concept    | 10        |
| rbtr__rbtr         | sql        | docstring  | 10        |
| rbtr__rbtr         | sql        | name       | 10        |
| rbtr__rbtr         | typescript | body       | 20        |
| rbtr__rbtr         | typescript | concept    | 10        |
| rbtr__rbtr         | typescript | docstring  | 11        |
| rbtr__rbtr         | typescript | name       | 10        |

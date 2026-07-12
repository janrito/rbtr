# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.020334 | 0.053841 | 0.925825 |
| identifier | 0.020334 | 0.033426 | 0.946239 |
| code       | 0.208055 | 0.742272 | 0.049673 |

## Result

| metric | current              | recommended         | delta                                |
| ------ | -------------------- | ------------------- | ------------------------------------ |
| MRR    | 3.999999938267203e-9 | 0.21031567797214318 | 0.21031567397214324 (+5257891930.4%) |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            |            | __all__    | 0.443817     | 0.427624 | -0.0162 | 0.471481            | 0.453808        | -0.0177          |
| __all__            | __all__    | __all__    | 0.463313     | 0.450549 | -0.0128 | 0.499615            | 0.490144        | -0.0095          |
| __all__            | __all__    | body       | 0.496739     | 0.470997 | -0.0257 | 0.541199            | 0.521351        | -0.0198          |
| __all__            | __all__    | concept    | 0.177647     | 0.171985 | -0.0057 | 0.210482            | 0.203077        | -0.0074          |
| __all__            | __all__    | docstring  | 0.695852     | 0.6804   | -0.0155 | 0.727141            | 0.719518        | -0.0076          |
| __all__            | __all__    | name       | 0.569728     | 0.570133 | 0.0004  | 0.600786            | 0.603167        | 0.0024           |
| __all__            | bash       | __all__    | 0.615487     | 0.629143 | 0.0137  | 0.668275            | 0.688075        | 0.0198           |
| __all__            | css        | __all__    | 0.697159     | 0.666717 | -0.0304 | 0.730645            | 0.710267        | -0.0204          |
| __all__            | html       | __all__    | 0.3234375    | 0.262326 | -0.0611 | 0.343175            | 0.29723         | -0.0459          |
| __all__            | javascript | __all__    | 0.570713     | 0.544099 | -0.0266 | 0.606333            | 0.579399        | -0.0269          |
| __all__            | json       | __all__    | 0.180471     | 0.164526 | -0.0159 | 0.210442            | 0.193627        | -0.0168          |
| __all__            | markdown   | __all__    | 0.205274     | 0.176338 | -0.0289 | 0.225527            | 0.201417        | -0.0241          |
| __all__            | python     | __all__    | 0.598675     | 0.606921 | 0.0082  | 0.642819            | 0.648053        | 0.0052           |
| __all__            | rst        | __all__    | 0.004608     | 0.004608 | 0.0     | 0.010753            | 0.010753        | 0.0              |
| __all__            | rust       | __all__    | 0.581184     | 0.574102 | -0.0071 | 0.640342            | 0.635493        | -0.0048          |
| __all__            | sql        | __all__    | 0.62284      | 0.625926 | 0.0031  | 0.68853             | 0.695887        | 0.0074           |
| __all__            | toml       | __all__    | 0.167907     | 0.104621 | -0.0633 | 0.191943            | 0.157037        | -0.0349          |
| __all__            | typescript | __all__    | 0.582346     | 0.58305  | 0.0007  | 0.625416            | 0.622674        | -0.0027          |
| __all__            | yaml       | __all__    | 0.14569      | 0.143166 | -0.0025 | 0.173153            | 0.183485        | 0.0103           |
| anthropics__skills | __all__    | __all__    | 0.579295     | 0.547484 | -0.0318 | 0.629008            | 0.598532        | -0.0305          |
| astral-sh__uv      | __all__    | __all__    | 0.453504     | 0.447944 | -0.0056 | 0.484416            | 0.486699        | 0.0023           |
| badlogic__pi-mono  | __all__    | __all__    | 0.484073     | 0.478738 | -0.0053 | 0.521001            | 0.5178          | -0.0032          |
| django__django     | __all__    | __all__    | 0.347153     | 0.334263 | -0.0129 | 0.36912             | 0.36045         | -0.0087          |
| rbtr__rbtr         | __all__    | __all__    | 0.47081      | 0.464956 | -0.0059 | 0.519481            | 0.511588        | -0.0079          |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.concept]
alpha = 0.02033447162493074
beta = 0.053840606361226506
gamma = 0.9258249220138428

[search_weights.identifier]
alpha = 0.02033447162493074
beta = 0.03342632797195456
gamma = 0.9462392004031146

[search_weights.code]
alpha = 0.20805514624356541
beta = 0.7422719142866617
gamma = 0.04967293946977288
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 1554
- elapsed: 8587 s

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

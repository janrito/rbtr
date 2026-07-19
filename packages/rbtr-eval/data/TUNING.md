# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.020334 | 0.033426 | 0.946239 |
| identifier | 0.007601 | 0.092751 | 0.899648 |
| code       | 0.063497 | 0.611428 | 0.325074 |

## Result

| metric | current             | recommended         | delta                       |
| ------ | ------------------- | ------------------- | --------------------------- |
| MRR    | 0.21282946588951207 | 0.21913559881227318 | 0.00630613292276111 (+3.0%) |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            |            | __all__    | 0.440261     | 0.417655 | -0.0226 | 0.467426            | 0.442322        | -0.0251          |
| __all__            | __all__    | __all__    | 0.470568     | 0.474809 | 0.0042  | 0.507477            | 0.511264        | 0.0038           |
| __all__            | __all__    | body       | 0.496447     | 0.509966 | 0.0135  | 0.540575            | 0.552852        | 0.0123           |
| __all__            | __all__    | concept    | 0.205394     | 0.194362 | -0.011  | 0.241091            | 0.231158        | -0.0099          |
| __all__            | __all__    | docstring  | 0.695852     | 0.691774 | -0.0041 | 0.727141            | 0.722388        | -0.0048          |
| __all__            | __all__    | name       | 0.569728     | 0.581262 | 0.0115  | 0.600786            | 0.611546        | 0.0108           |
| __all__            | bash       | __all__    | 0.653598     | 0.668605 | 0.015   | 0.698899            | 0.716746        | 0.0178           |
| __all__            | css        | __all__    | 0.702609     | 0.69045  | -0.0122 | 0.740107            | 0.729528        | -0.0106          |
| __all__            | html       | __all__    | 0.3234375    | 0.340402 | 0.017   | 0.343175            | 0.356092        | 0.0129           |
| __all__            | javascript | __all__    | 0.578999     | 0.588777 | 0.0098  | 0.618821            | 0.626371        | 0.0075           |
| __all__            | json       | __all__    | 0.183063     | 0.179349 | -0.0037 | 0.214143            | 0.208591        | -0.0056          |
| __all__            | markdown   | __all__    | 0.20419      | 0.237513 | 0.0333  | 0.224934            | 0.260658        | 0.0357           |
| __all__            | python     | __all__    | 0.611332     | 0.624582 | 0.0132  | 0.659315            | 0.667263        | 0.0079           |
| __all__            | rst        | __all__    | 0.004608     | 0.009217 | 0.0046  | 0.010753            | 0.021505        | 0.0108           |
| __all__            | rust       | __all__    | 0.605144     | 0.591997 | -0.0131 | 0.655507            | 0.645857        | -0.0096          |
| __all__            | sql        | __all__    | 0.555247     | 0.583272 | 0.028   | 0.615139            | 0.646381        | 0.0312           |
| __all__            | toml       | __all__    | 0.167907     | 0.138021 | -0.0299 | 0.191943            | 0.155627        | -0.0363          |
| __all__            | typescript | __all__    | 0.589446     | 0.57601  | -0.0134 | 0.633182            | 0.621295        | -0.0119          |
| __all__            | yaml       | __all__    | 0.147426     | 0.170194 | 0.0228  | 0.177535            | 0.198715        | 0.0212           |
| anthropics__skills | __all__    | __all__    | 0.594056     | 0.588957 | -0.0051 | 0.643834            | 0.637386        | -0.0064          |
| astral-sh__uv      | __all__    | __all__    | 0.470873     | 0.472458 | 0.0016  | 0.500893            | 0.50177         | 0.0009           |
| badlogic__pi-mono  | __all__    | __all__    | 0.479128     | 0.481397 | 0.0023  | 0.518472            | 0.522615        | 0.0041           |
| django__django     | __all__    | __all__    | 0.349083     | 0.355717 | 0.0066  | 0.373394            | 0.379049        | 0.0057           |
| rbtr__rbtr         | __all__    | __all__    | 0.469803     | 0.492147 | 0.0223  | 0.517351            | 0.539036        | 0.0217           |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.concept]
alpha = 0.02033447162493074
beta = 0.03342632797195456
gamma = 0.9462392004031146

[search_weights.identifier]
alpha = 0.0076011360583828486
beta = 0.09275092124464959
gamma = 0.8996479426969676

[search_weights.code]
alpha = 0.0634972115247005
beta = 0.6114282980612943
gamma = 0.3250744904140052
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 1552
- elapsed: 8782 s

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills |            | body       | 10        |
| anthropics__skills |            | concept    | 12        |
| anthropics__skills |            | name       | 10        |
| anthropics__skills | bash       | body       | 15        |
| anthropics__skills | bash       | concept    | 10        |
| anthropics__skills | bash       | docstring  | 6         |
| anthropics__skills | bash       | name       | 10        |
| anthropics__skills | css        | body       | 11        |
| anthropics__skills | css        | concept    | 12        |
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
| anthropics__skills | python     | concept    | 11        |
| anthropics__skills | python     | docstring  | 13        |
| anthropics__skills | python     | name       | 10        |
| anthropics__skills | typescript | body       | 20        |
| anthropics__skills | typescript | concept    | 11        |
| anthropics__skills | typescript | docstring  | 9         |
| anthropics__skills | typescript | name       | 10        |
| astral-sh__uv      |            | body       | 10        |
| astral-sh__uv      |            | concept    | 10        |
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
| astral-sh__uv      | python     | concept    | 11        |
| astral-sh__uv      | python     | docstring  | 14        |
| astral-sh__uv      | python     | name       | 10        |
| astral-sh__uv      | rust       | body       | 20        |
| astral-sh__uv      | rust       | concept    | 12        |
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
| badlogic__pi-mono  | markdown   | concept    | 10        |
| badlogic__pi-mono  | markdown   | name       | 11        |
| badlogic__pi-mono  | typescript | body       | 20        |
| badlogic__pi-mono  | typescript | concept    | 12        |
| badlogic__pi-mono  | typescript | docstring  | 13        |
| badlogic__pi-mono  | typescript | name       | 10        |
| django__django     |            | body       | 10        |
| django__django     |            | concept    | 12        |
| django__django     |            | name       | 10        |
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
| rbtr__rbtr         | python     | concept    | 10        |
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

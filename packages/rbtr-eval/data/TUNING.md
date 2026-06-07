# rbtr search-weight tuning report

## Fusion weights

The search score fuses three channels:

- __alpha__ — semantic (embedding cosine similarity)
- __beta__ — lexical (BM25 keyword search)
- __gamma__ — name-match (identifier matching)

`score = alpha × semantic + beta × lexical + gamma × name`

| kind       | alpha    | beta     | gamma    |
| ---------- | -------- | -------- | -------- |
| concept    | 0.568045 | 0.399817 | 0.032139 |
| identifier | 0.000206 | 0.300232 | 0.699563 |
| code       | 0.00008  | 0.475412 | 0.524508 |

## Result

| metric | current            | recommended        | delta                        |
| ------ | ------------------ | ------------------ | ---------------------------- |
| MRR    | 0.2967358488152385 | 0.3000743572091827 | 0.003338508393944173 (+1.1%) |

## Impact by dimension

MRR comparison between current (baseline) and recommended
(best) weights, broken down by repo, language, and query
kind.

| slug               | language   | provenance | baseline_mrr | best_mrr | delta   | baseline_ndcg_at_10 | best_ndcg_at_10 | delta_ndcg_at_10 |
| ------------------ | ---------- | ---------- | ------------ | -------- | ------- | ------------------- | --------------- | ---------------- |
| __all__            | __all__    | __all__    | 0.389849     | 0.425015 | 0.0352  | 0.425369            | 0.459039        | 0.0337           |
| __all__            | __all__    | body       | 0.424729     | 0.527779 | 0.1031  | 0.473218            | 0.572102        | 0.0989           |
| __all__            | __all__    | concept    | 0.188376     | 0.190376 | 0.002   | 0.20789             | 0.209345        | 0.0015           |
| __all__            | __all__    | docstring  | 0.69858      | 0.708086 | 0.0095  | 0.749795            | 0.757098        | 0.0073           |
| __all__            | __all__    | name       | 0.4453       | 0.454984 | 0.0097  | 0.478206            | 0.488368        | 0.0102           |
| __all__            | css        | __all__    | 0.545437     | 0.647222 | 0.1018  | 0.571248            | 0.652182        | 0.0809           |
| __all__            | html       | __all__    | 0.3          | 0.365278 | 0.0653  | 0.323085            | 0.390275        | 0.0672           |
| __all__            | javascript | __all__    | 0.574896     | 0.565848 | -0.009  | 0.602961            | 0.596042        | -0.0069          |
| __all__            | json       | __all__    | 0.153366     | 0.17744  | 0.0241  | 0.17019             | 0.198517        | 0.0283           |
| __all__            | markdown   | __all__    | 0.155839     | 0.222034 | 0.0662  | 0.196069            | 0.258543        | 0.0625           |
| __all__            | python     | __all__    | 0.628242     | 0.624692 | -0.0035 | 0.675443            | 0.671095        | -0.0043          |
| __all__            | rst        | __all__    | 0.041799     | 0.045132 | 0.0033  | 0.054479            | 0.072843        | 0.0184           |
| __all__            | rust       | __all__    | 0.630208     | 0.631349 | 0.0011  | 0.684634            | 0.685306        | 0.0007           |
| __all__            | toml       | __all__    | 0.125556     | 0.242593 | 0.117   | 0.175016            | 0.272699        | 0.0977           |
| __all__            | typescript | __all__    | 0.789459     | 0.790243 | 0.0008  | 0.834464            | 0.837659        | 0.0032           |
| __all__            | yaml       | __all__    | 0.067857     | 0.171653 | 0.1038  | 0.102902            | 0.205006        | 0.1021           |
| anthropics__skills | __all__    | __all__    | 0.546757     | 0.586502 | 0.0397  | 0.58613             | 0.624351        | 0.0382           |
| astral-sh__uv      | __all__    | __all__    | 0.327054     | 0.382476 | 0.0554  | 0.362794            | 0.413187        | 0.0504           |
| badlogic__pi-mono  | __all__    | __all__    | 0.495812     | 0.537824 | 0.042   | 0.52808             | 0.564272        | 0.0362           |
| django__django     | __all__    | __all__    | 0.260861     | 0.280818 | 0.02    | 0.297239            | 0.321116        | 0.0239           |
| rbtr__rbtr         | __all__    | __all__    | 0.511984     | 0.535853 | 0.0239  | 0.547617            | 0.570245        | 0.0226           |

## Convergence

![Convergence](convergence.png)

## Simplex exploration

![Simplex](simplex.png)

## Recommended config

```toml
[search_weights.concept]
alpha = 0.5680445610939323
beta = 0.3998165021436872
gamma = 0.03213893676238051

[search_weights.identifier]
alpha = 0.00020556262208889717
beta = 0.30023168487679514
gamma = 0.6995627525011159

[search_weights.code]
alpha = 8.015780612902201e-05
beta = 0.4754115709036819
gamma = 0.5245082712901891
```

## Run metadata

- Optuna trials: 50
- queries evaluated: 840
- elapsed: 5277 s

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

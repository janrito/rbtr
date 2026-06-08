# Reranker tuning report

## Parameters

- **`reranker_pool`** — how many fusion candidates enter
  the cross-encoder reranker. Larger pools give the
  reranker more candidates to promote but cost more
  latency.
- **`reranker_blend_weight`** — blend between fusion and
  reranker scores: `score = w * fusion + (1 - w) * reranker`.
  `0.0` = pure reranker, `1.0` = pure fusion (no reranking
  effect).

## Grid results

| pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % | hit@10 % | miss % | p50 ms | p95 ms |
| ---- | ----- | --------- | ------ | ------- | ------- | ------- | -------- | ------ | ------ | ------ |
| 80   | 0.25  | 416       | 0.5672 | 0.6032  | 49.3    | 62.3    | 71.6     | 28.4   | 7665   | 13473  |
| 50   | 0.25  | 416       | 0.5581 | 0.5907  | 48.8    | 61.3    | 69.2     | 30.8   | 5481   | 9349   |
| 80   | 0.0   | 416       | 0.5519 | 0.592   | 46.2    | 62.5    | 71.6     | 28.4   | 7665   | 13473  |
| 80   | 0.5   | 416       | 0.5516 | 0.5911  | 47.4    | 61.1    | 71.6     | 28.4   | 7665   | 13473  |
| 50   | 0.5   | 416       | 0.5429 | 0.579   | 46.9    | 60.1    | 69.2     | 30.8   | 5481   | 9349   |
| 50   | 0.0   | 416       | 0.5357 | 0.574   | 45.0    | 60.6    | 69.2     | 30.8   | 5481   | 9349   |
| 80   | 0.75  | 416       | 0.533  | 0.5764  | 45.9    | 56.5    | 71.6     | 28.4   | 7665   | 13473  |
| 20   | 0.25  | 416       | 0.5276 | 0.556   | 46.4    | 58.2    | 64.4     | 35.6   | 2422   | 4466   |
| 50   | 0.75  | 416       | 0.5268 | 0.5662  | 45.7    | 56.0    | 69.2     | 30.8   | 5481   | 9349   |
| 20   | 0.5   | 416       | 0.5168 | 0.5478  | 45.0    | 57.2    | 64.4     | 35.6   | 2422   | 4466   |
| 20   | 0.75  | 416       | 0.5069 | 0.5399  | 44.2    | 54.3    | 64.4     | 35.6   | 2422   | 4466   |
| 80   | 1.0   | 416       | 0.506  | 0.5555  | 42.5    | 54.6    | 71.6     | 28.4   | 7665   | 13473  |
| 20   | 0.0   | 416       | 0.5048 | 0.5393  | 42.1    | 58.7    | 64.4     | 35.6   | 2422   | 4466   |
| 50   | 1.0   | 416       | 0.4878 | 0.5361  | 40.6    | 52.9    | 69.2     | 30.8   | 5481   | 9349   |
| 20   | 1.0   | 416       | 0.4631 | 0.5061  | 38.7    | 50.5    | 64.4     | 35.6   | 2422   | 4466   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 121       | 0.6635 | 0.7049  | 57.9    | 71.9    |
| body       | 50   | 0.25  | 121       | 0.6619 | 0.7017  | 57.9    | 71.9    |
| body       | 80   | 0.75  | 121       | 0.6613 | 0.7024  | 58.7    | 70.2    |
| body       | 50   | 0.75  | 121       | 0.66   | 0.6996  | 58.7    | 70.2    |
| body       | 80   | 0.5   | 121       | 0.6556 | 0.6985  | 57.0    | 71.1    |
| body       | 50   | 0.5   | 121       | 0.6544 | 0.6957  | 57.0    | 71.1    |
| body       | 80   | 0.0   | 121       | 0.6387 | 0.6862  | 53.7    | 71.9    |
| body       | 50   | 0.0   | 121       | 0.6346 | 0.681   | 53.7    | 71.1    |
| body       | 20   | 0.25  | 121       | 0.6314 | 0.6687  | 55.4    | 68.6    |
| body       | 20   | 0.75  | 121       | 0.6302 | 0.6669  | 56.2    | 66.9    |
| body       | 20   | 0.5   | 121       | 0.6253 | 0.6638  | 54.5    | 68.6    |
| body       | 80   | 1.0   | 121       | 0.6133 | 0.6659  | 52.1    | 66.1    |
| body       | 20   | 0.0   | 121       | 0.613  | 0.6549  | 52.1    | 69.4    |
| body       | 50   | 1.0   | 121       | 0.5871 | 0.6443  | 47.9    | 63.6    |
| body       | 20   | 1.0   | 121       | 0.5575 | 0.6118  | 45.5    | 62.0    |
| concept    | 80   | 0.0   | 125       | 0.4468 | 0.4749  | 37.6    | 51.2    |
| concept    | 50   | 0.0   | 125       | 0.4003 | 0.4244  | 33.6    | 46.4    |
| concept    | 80   | 0.25  | 125       | 0.3541 | 0.4038  | 24.8    | 44.0    |
| concept    | 80   | 0.5   | 125       | 0.3296 | 0.3843  | 23.2    | 40.8    |
| concept    | 50   | 0.25  | 125       | 0.3279 | 0.369   | 23.2    | 41.6    |
| concept    | 20   | 0.0   | 125       | 0.3193 | 0.3395  | 26.4    | 37.6    |
| concept    | 50   | 0.5   | 125       | 0.3043 | 0.3503  | 21.6    | 38.4    |
| concept    | 20   | 0.25  | 125       | 0.2718 | 0.3034  | 19.2    | 35.2    |
| concept    | 80   | 0.75  | 125       | 0.2642 | 0.3329  | 16.8    | 27.2    |
| concept    | 20   | 0.5   | 125       | 0.257  | 0.2918  | 18.4    | 32.8    |
| concept    | 80   | 1.0   | 125       | 0.2499 | 0.3209  | 16.0    | 26.4    |
| concept    | 50   | 0.75  | 125       | 0.2474 | 0.3056  | 16.0    | 26.4    |
| concept    | 50   | 1.0   | 125       | 0.234  | 0.2939  | 16.0    | 24.8    |
| concept    | 20   | 0.75  | 125       | 0.2231 | 0.265   | 15.2    | 24.0    |
| concept    | 20   | 1.0   | 125       | 0.2031 | 0.2484  | 14.4    | 20.8    |
| docstring  | 80   | 0.25  | 45        | 0.8722 | 0.8929  | 82.2    | 91.1    |
| docstring  | 50   | 0.25  | 45        | 0.8667 | 0.8833  | 82.2    | 91.1    |
| docstring  | 80   | 0.5   | 45        | 0.86   | 0.8837  | 80.0    | 91.1    |
| docstring  | 50   | 0.5   | 45        | 0.8556 | 0.8751  | 80.0    | 91.1    |
| docstring  | 80   | 0.75  | 45        | 0.8533 | 0.8783  | 80.0    | 88.9    |
| docstring  | 50   | 0.75  | 45        | 0.8481 | 0.869   | 80.0    | 88.9    |
| docstring  | 80   | 0.0   | 45        | 0.8322 | 0.8637  | 73.3    | 91.1    |
| docstring  | 50   | 0.0   | 45        | 0.8241 | 0.8522  | 73.3    | 91.1    |
| docstring  | 20   | 0.25  | 45        | 0.8222 | 0.8389  | 77.8    | 86.7    |
| docstring  | 20   | 0.5   | 45        | 0.8222 | 0.8389  | 77.8    | 86.7    |
| docstring  | 80   | 1.0   | 45        | 0.8207 | 0.8535  | 75.6    | 86.7    |
| docstring  | 20   | 0.75  | 45        | 0.8111 | 0.8307  | 75.6    | 86.7    |
| docstring  | 50   | 1.0   | 45        | 0.7836 | 0.8196  | 71.1    | 84.4    |
| docstring  | 20   | 0.0   | 45        | 0.7815 | 0.8093  | 68.9    | 88.9    |
| docstring  | 20   | 1.0   | 45        | 0.7541 | 0.7866  | 68.9    | 82.2    |
| name       | 80   | 0.25  | 125       | 0.5774 | 0.6     | 53.6    | 60.8    |
| name       | 20   | 0.25  | 125       | 0.5767 | 0.5977  | 53.6    | 60.8    |
| name       | 50   | 0.25  | 125       | 0.5767 | 0.5994  | 53.6    | 60.0    |
| name       | 80   | 0.75  | 125       | 0.5622 | 0.5891  | 50.4    | 60.8    |
| name       | 20   | 0.75  | 125       | 0.5619 | 0.5872  | 50.4    | 60.8    |
| name       | 80   | 0.5   | 125       | 0.5619 | 0.5886  | 50.4    | 60.8    |
| name       | 50   | 0.75  | 125       | 0.5615 | 0.5886  | 50.4    | 60.0    |
| name       | 20   | 0.5   | 125       | 0.5615 | 0.5865  | 50.4    | 60.0    |
| name       | 50   | 0.5   | 125       | 0.5612 | 0.588   | 50.4    | 60.0    |
| name       | 80   | 1.0   | 125       | 0.545  | 0.5759  | 48.0    | 60.0    |
| name       | 50   | 1.0   | 125       | 0.539  | 0.5714  | 47.2    | 59.2    |
| name       | 20   | 1.0   | 125       | 0.527  | 0.5604  | 45.6    | 57.6    |
| name       | 20   | 0.0   | 125       | 0.486  | 0.5299  | 38.4    | 58.4    |
| name       | 80   | 0.0   | 125       | 0.4719 | 0.5202  | 37.6    | 54.4    |
| name       | 50   | 0.0   | 125       | 0.4717 | 0.52    | 37.6    | 53.6    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| code       | 80   | 0.25  | 121       | 0.6635 | 0.7049  | 57.9    | 71.9    |
| code       | 50   | 0.25  | 121       | 0.6619 | 0.7017  | 57.9    | 71.9    |
| code       | 80   | 0.75  | 121       | 0.6613 | 0.7024  | 58.7    | 70.2    |
| code       | 50   | 0.75  | 121       | 0.66   | 0.6996  | 58.7    | 70.2    |
| code       | 80   | 0.5   | 121       | 0.6556 | 0.6985  | 57.0    | 71.1    |
| code       | 50   | 0.5   | 121       | 0.6544 | 0.6957  | 57.0    | 71.1    |
| code       | 80   | 0.0   | 121       | 0.6387 | 0.6862  | 53.7    | 71.9    |
| code       | 50   | 0.0   | 121       | 0.6346 | 0.681   | 53.7    | 71.1    |
| code       | 20   | 0.25  | 121       | 0.6314 | 0.6687  | 55.4    | 68.6    |
| code       | 20   | 0.75  | 121       | 0.6302 | 0.6669  | 56.2    | 66.9    |
| code       | 20   | 0.5   | 121       | 0.6253 | 0.6638  | 54.5    | 68.6    |
| code       | 80   | 1.0   | 121       | 0.6133 | 0.6659  | 52.1    | 66.1    |
| code       | 20   | 0.0   | 121       | 0.613  | 0.6549  | 52.1    | 69.4    |
| code       | 50   | 1.0   | 121       | 0.5871 | 0.6443  | 47.9    | 63.6    |
| code       | 20   | 1.0   | 121       | 0.5575 | 0.6118  | 45.5    | 62.0    |
| concept    | 80   | 0.0   | 170       | 0.5488 | 0.5778  | 47.1    | 61.8    |
| concept    | 50   | 0.0   | 170       | 0.5125 | 0.5376  | 44.1    | 58.2    |
| concept    | 80   | 0.25  | 170       | 0.4913 | 0.5333  | 40.0    | 56.5    |
| concept    | 50   | 0.25  | 170       | 0.4705 | 0.5052  | 38.8    | 54.7    |
| concept    | 80   | 0.5   | 170       | 0.47   | 0.5165  | 38.2    | 54.1    |
| concept    | 50   | 0.5   | 170       | 0.4502 | 0.4893  | 37.1    | 52.4    |
| concept    | 20   | 0.0   | 170       | 0.4416 | 0.4639  | 37.6    | 51.2    |
| concept    | 80   | 0.75  | 170       | 0.4202 | 0.4773  | 33.5    | 43.5    |
| concept    | 20   | 0.25  | 170       | 0.4175 | 0.4452  | 34.7    | 48.8    |
| concept    | 20   | 0.5   | 170       | 0.4066 | 0.4366  | 34.1    | 47.1    |
| concept    | 50   | 0.75  | 170       | 0.4064 | 0.4547  | 32.9    | 42.9    |
| concept    | 80   | 1.0   | 170       | 0.401  | 0.4619  | 31.8    | 42.4    |
| concept    | 50   | 1.0   | 170       | 0.3795 | 0.4331  | 30.6    | 40.6    |
| concept    | 20   | 0.75  | 170       | 0.3788 | 0.4147  | 31.2    | 40.6    |
| concept    | 20   | 1.0   | 170       | 0.3489 | 0.3909  | 28.8    | 37.1    |
| identifier | 80   | 0.25  | 125       | 0.5774 | 0.6     | 53.6    | 60.8    |
| identifier | 20   | 0.25  | 125       | 0.5767 | 0.5977  | 53.6    | 60.8    |
| identifier | 50   | 0.25  | 125       | 0.5767 | 0.5994  | 53.6    | 60.0    |
| identifier | 80   | 0.75  | 125       | 0.5622 | 0.5891  | 50.4    | 60.8    |
| identifier | 20   | 0.75  | 125       | 0.5619 | 0.5872  | 50.4    | 60.8    |
| identifier | 80   | 0.5   | 125       | 0.5619 | 0.5886  | 50.4    | 60.8    |
| identifier | 50   | 0.75  | 125       | 0.5615 | 0.5886  | 50.4    | 60.0    |
| identifier | 20   | 0.5   | 125       | 0.5615 | 0.5865  | 50.4    | 60.0    |
| identifier | 50   | 0.5   | 125       | 0.5612 | 0.588   | 50.4    | 60.0    |
| identifier | 80   | 1.0   | 125       | 0.545  | 0.5759  | 48.0    | 60.0    |
| identifier | 50   | 1.0   | 125       | 0.539  | 0.5714  | 47.2    | 59.2    |
| identifier | 20   | 1.0   | 125       | 0.527  | 0.5604  | 45.6    | 57.6    |
| identifier | 20   | 0.0   | 125       | 0.486  | 0.5299  | 38.4    | 58.4    |
| identifier | 80   | 0.0   | 125       | 0.4719 | 0.5202  | 37.6    | 54.4    |
| identifier | 50   | 0.0   | 125       | 0.4717 | 0.52    | 37.6    | 53.6    |

## Per-kind recommendation

| kind       | pool | blend | n_queries | MRR    | NDCG@10 |
| ---------- | ---- | ----- | --------- | ------ | ------- |
| code       | 80   | 0.25  | 121       | 0.6635 | 0.7049  |
| concept    | 80   | 0.0   | 170       | 0.5488 | 0.5778  |
| identifier | 80   | 0.25  | 125       | 0.5774 | 0.6     |

```toml
[reranker_settings.code]
pool = 80
blend_weight = 0.25

[reranker_settings.concept]
pool = 80
blend_weight = 0.0

[reranker_settings.identifier]
pool = 80
blend_weight = 0.25
```

## Per-slug breakdown (best config)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 35        | 0.7762 | 0.8248  | 65.7    | 91.4    |
| astral-sh__uv      | 100       | 0.4861 | 0.5294  | 42.0    | 50.0    |
| badlogic__pi-mono  | 81        | 0.6224 | 0.6433  | 55.6    | 69.1    |
| django__django     | 130       | 0.4864 | 0.5265  | 40.8    | 54.6    |
| rbtr__rbtr         | 70        | 0.6651 | 0.6941  | 60.0    | 71.4    |

## Per-language breakdown (best config)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| css        | 30        | 0.7083 | 0.7318  | 63.3    | 76.7    |
| html       | 15        | 0.4622 | 0.5275  | 33.3    | 53.3    |
| javascript | 40        | 0.9    | 0.9131  | 85.0    | 95.0    |
| json       | 60        | 0.2043 | 0.2446  | 16.7    | 18.3    |
| markdown   | 71        | 0.4571 | 0.4998  | 36.6    | 53.5    |
| python     | 80        | 0.7907 | 0.8236  | 71.2    | 87.5    |
| rst        | 15        | 0.2994 | 0.3701  | 20.0    | 26.7    |
| rust       | 20        | 0.6017 | 0.6749  | 45.0    | 70.0    |
| toml       | 15        | 0.4333 | 0.4421  | 40.0    | 46.7    |
| typescript | 40        | 0.8327 | 0.8622  | 75.0    | 92.5    |
| yaml       | 30        | 0.2497 | 0.2778  | 20.0    | 30.0    |

## Recommended config (flat)

```toml
[reranker]
pool = 80
blend_weight = 0.25
```

## Run metadata

| field             | value  |
| ----------------- | ------ |
| queries evaluated | 420    |
| configs evaluated | 15     |
| elapsed           | 6762 s |

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills | markdown   | body       | 5         |
| anthropics__skills | markdown   | concept    | 5         |
| anthropics__skills | markdown   | name       | 5         |
| anthropics__skills | python     | body       | 5         |
| anthropics__skills | python     | concept    | 5         |
| anthropics__skills | python     | docstring  | 5         |
| anthropics__skills | python     | name       | 5         |
| astral-sh__uv      | json       | body       | 5         |
| astral-sh__uv      | json       | concept    | 5         |
| astral-sh__uv      | json       | name       | 5         |
| astral-sh__uv      | markdown   | body       | 5         |
| astral-sh__uv      | markdown   | concept    | 5         |
| astral-sh__uv      | markdown   | name       | 5         |
| astral-sh__uv      | python     | body       | 5         |
| astral-sh__uv      | python     | concept    | 5         |
| astral-sh__uv      | python     | docstring  | 5         |
| astral-sh__uv      | python     | name       | 5         |
| astral-sh__uv      | rust       | body       | 5         |
| astral-sh__uv      | rust       | concept    | 5         |
| astral-sh__uv      | rust       | docstring  | 5         |
| astral-sh__uv      | rust       | name       | 5         |
| astral-sh__uv      | toml       | body       | 5         |
| astral-sh__uv      | toml       | concept    | 5         |
| astral-sh__uv      | toml       | name       | 5         |
| astral-sh__uv      | yaml       | body       | 5         |
| astral-sh__uv      | yaml       | concept    | 5         |
| astral-sh__uv      | yaml       | name       | 5         |
| badlogic__pi-mono  | css        | body       | 5         |
| badlogic__pi-mono  | css        | concept    | 5         |
| badlogic__pi-mono  | css        | name       | 5         |
| badlogic__pi-mono  | javascript | body       | 5         |
| badlogic__pi-mono  | javascript | concept    | 5         |
| badlogic__pi-mono  | javascript | docstring  | 5         |
| badlogic__pi-mono  | javascript | name       | 5         |
| badlogic__pi-mono  | json       | body       | 5         |
| badlogic__pi-mono  | json       | concept    | 5         |
| badlogic__pi-mono  | json       | name       | 5         |
| badlogic__pi-mono  | markdown   | body       | 5         |
| badlogic__pi-mono  | markdown   | concept    | 5         |
| badlogic__pi-mono  | markdown   | name       | 5         |
| badlogic__pi-mono  | typescript | body       | 5         |
| badlogic__pi-mono  | typescript | concept    | 5         |
| badlogic__pi-mono  | typescript | docstring  | 5         |
| badlogic__pi-mono  | typescript | name       | 5         |
| django__django     | css        | body       | 5         |
| django__django     | css        | concept    | 5         |
| django__django     | css        | name       | 5         |
| django__django     | html       | body       | 5         |
| django__django     | html       | concept    | 5         |
| django__django     | html       | name       | 5         |
| django__django     | javascript | body       | 5         |
| django__django     | javascript | concept    | 5         |
| django__django     | javascript | docstring  | 5         |
| django__django     | javascript | name       | 5         |
| django__django     | json       | body       | 5         |
| django__django     | json       | concept    | 5         |
| django__django     | json       | name       | 5         |
| django__django     | markdown   | body       | 5         |
| django__django     | markdown   | concept    | 5         |
| django__django     | markdown   | name       | 5         |
| django__django     | python     | body       | 5         |
| django__django     | python     | concept    | 5         |
| django__django     | python     | docstring  | 5         |
| django__django     | python     | name       | 5         |
| django__django     | rst        | body       | 5         |
| django__django     | rst        | concept    | 5         |
| django__django     | rst        | name       | 5         |
| django__django     | yaml       | body       | 5         |
| django__django     | yaml       | concept    | 5         |
| django__django     | yaml       | name       | 5         |
| rbtr__rbtr         | json       | body       | 5         |
| rbtr__rbtr         | json       | concept    | 5         |
| rbtr__rbtr         | json       | name       | 5         |
| rbtr__rbtr         | markdown   | body       | 5         |
| rbtr__rbtr         | markdown   | concept    | 5         |
| rbtr__rbtr         | markdown   | name       | 5         |
| rbtr__rbtr         | python     | body       | 5         |
| rbtr__rbtr         | python     | concept    | 5         |
| rbtr__rbtr         | python     | docstring  | 5         |
| rbtr__rbtr         | python     | name       | 5         |
| rbtr__rbtr         | typescript | body       | 5         |
| rbtr__rbtr         | typescript | concept    | 5         |
| rbtr__rbtr         | typescript | docstring  | 5         |
| rbtr__rbtr         | typescript | name       | 5         |

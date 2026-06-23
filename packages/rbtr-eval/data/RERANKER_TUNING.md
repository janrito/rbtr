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
| 80   | 0.25  | 416       | 0.5606 | 0.5986  | 48.6    | 61.1    | 71.9     | 28.1   | 7615   | 13197  |
| 50   | 0.25  | 416       | 0.5494 | 0.5838  | 47.8    | 60.1    | 69.2     | 30.8   | 5333   | 8855   |
| 80   | 0.5   | 416       | 0.5482 | 0.5888  | 47.1    | 60.1    | 71.9     | 28.1   | 7615   | 13197  |
| 80   | 0.0   | 416       | 0.5454 | 0.5877  | 45.2    | 62.0    | 71.9     | 28.1   | 7615   | 13197  |
| 50   | 0.5   | 416       | 0.5378 | 0.5748  | 46.4    | 59.1    | 69.2     | 30.8   | 5333   | 8855   |
| 50   | 0.0   | 416       | 0.527  | 0.5673  | 43.8    | 60.1    | 69.2     | 30.8   | 5333   | 8855   |
| 80   | 0.75  | 416       | 0.525  | 0.5705  | 45.0    | 55.8    | 71.9     | 28.1   | 7615   | 13197  |
| 20   | 0.25  | 416       | 0.5214 | 0.5508  | 45.7    | 57.2    | 64.2     | 35.8   | 2343   | 4292   |
| 50   | 0.75  | 416       | 0.5195 | 0.5605  | 44.7    | 55.5    | 69.2     | 30.8   | 5333   | 8855   |
| 20   | 0.5   | 416       | 0.5136 | 0.5447  | 44.7    | 56.5    | 64.2     | 35.8   | 2343   | 4292   |
| 20   | 0.75  | 416       | 0.5007 | 0.5347  | 43.3    | 54.3    | 64.2     | 35.8   | 2343   | 4292   |
| 80   | 1.0   | 416       | 0.4975 | 0.5493  | 41.3    | 53.8    | 71.9     | 28.1   | 7615   | 13197  |
| 20   | 0.0   | 416       | 0.4974 | 0.5331  | 41.1    | 57.9    | 64.2     | 35.8   | 2343   | 4292   |
| 50   | 1.0   | 416       | 0.4823 | 0.5316  | 39.9    | 52.6    | 69.2     | 30.8   | 5333   | 8855   |
| 20   | 1.0   | 416       | 0.4552 | 0.4992  | 38.0    | 49.0    | 64.2     | 35.8   | 2343   | 4292   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 121       | 0.66   | 0.7041  | 57.0    | 71.9    |
| body       | 50   | 0.25  | 121       | 0.6582 | 0.7008  | 57.0    | 71.9    |
| body       | 80   | 0.5   | 121       | 0.6564 | 0.7009  | 57.0    | 71.1    |
| body       | 50   | 0.5   | 121       | 0.6551 | 0.698   | 57.0    | 71.1    |
| body       | 50   | 0.75  | 121       | 0.6534 | 0.6967  | 57.0    | 70.2    |
| body       | 80   | 0.75  | 121       | 0.6524 | 0.6973  | 57.0    | 70.2    |
| body       | 20   | 0.25  | 121       | 0.6407 | 0.6817  | 55.4    | 70.2    |
| body       | 20   | 0.5   | 121       | 0.6389 | 0.6801  | 55.4    | 70.2    |
| body       | 20   | 0.75  | 121       | 0.6338 | 0.6761  | 54.5    | 69.4    |
| body       | 80   | 0.0   | 121       | 0.6285 | 0.6804  | 52.1    | 71.9    |
| body       | 50   | 0.0   | 121       | 0.6238 | 0.6746  | 52.1    | 71.1    |
| body       | 20   | 0.0   | 121       | 0.619  | 0.6653  | 52.1    | 70.2    |
| body       | 80   | 1.0   | 121       | 0.5972 | 0.6557  | 48.8    | 66.1    |
| body       | 50   | 1.0   | 121       | 0.5807 | 0.6415  | 46.3    | 64.5    |
| body       | 20   | 1.0   | 121       | 0.5508 | 0.6121  | 43.8    | 61.2    |
| concept    | 80   | 0.0   | 125       | 0.4408 | 0.4704  | 36.8    | 50.4    |
| concept    | 50   | 0.0   | 125       | 0.3863 | 0.4118  | 32.0    | 44.8    |
| concept    | 80   | 0.25  | 125       | 0.3508 | 0.4006  | 25.6    | 40.8    |
| concept    | 80   | 0.5   | 125       | 0.3337 | 0.3866  | 24.8    | 38.4    |
| concept    | 50   | 0.25  | 125       | 0.3167 | 0.358   | 23.2    | 37.6    |
| concept    | 50   | 0.5   | 125       | 0.3006 | 0.3449  | 22.4    | 35.2    |
| concept    | 20   | 0.0   | 125       | 0.2937 | 0.3143  | 24.0    | 34.4    |
| concept    | 80   | 0.75  | 125       | 0.2714 | 0.3373  | 19.2    | 26.4    |
| concept    | 20   | 0.25  | 125       | 0.2561 | 0.2853  | 19.2    | 30.4    |
| concept    | 80   | 1.0   | 125       | 0.2547 | 0.3236  | 17.6    | 25.6    |
| concept    | 50   | 0.75  | 125       | 0.2538 | 0.3076  | 18.4    | 25.6    |
| concept    | 20   | 0.5   | 125       | 0.2493 | 0.2797  | 19.2    | 28.8    |
| concept    | 50   | 1.0   | 125       | 0.2399 | 0.2959  | 17.6    | 24.0    |
| concept    | 20   | 0.75  | 125       | 0.2219 | 0.2581  | 16.8    | 23.2    |
| concept    | 20   | 1.0   | 125       | 0.2049 | 0.2438  | 16.0    | 20.0    |
| docstring  | 80   | 0.25  | 45        | 0.8611 | 0.8847  | 80.0    | 91.1    |
| docstring  | 50   | 0.25  | 45        | 0.8556 | 0.8751  | 80.0    | 91.1    |
| docstring  | 80   | 0.5   | 45        | 0.8378 | 0.8673  | 75.6    | 91.1    |
| docstring  | 50   | 0.5   | 45        | 0.8333 | 0.8587  | 75.6    | 91.1    |
| docstring  | 80   | 0.0   | 45        | 0.8322 | 0.8637  | 73.3    | 91.1    |
| docstring  | 50   | 0.0   | 45        | 0.8241 | 0.8522  | 73.3    | 91.1    |
| docstring  | 80   | 0.75  | 45        | 0.8193 | 0.853   | 73.3    | 88.9    |
| docstring  | 50   | 0.75  | 45        | 0.8148 | 0.8444  | 73.3    | 88.9    |
| docstring  | 20   | 0.25  | 45        | 0.8111 | 0.8307  | 75.6    | 86.7    |
| docstring  | 80   | 1.0   | 45        | 0.8076 | 0.8435  | 73.3    | 86.7    |
| docstring  | 20   | 0.5   | 45        | 0.8    | 0.8225  | 73.3    | 86.7    |
| docstring  | 20   | 0.75  | 45        | 0.7889 | 0.8143  | 71.1    | 86.7    |
| docstring  | 20   | 0.0   | 45        | 0.7815 | 0.8093  | 68.9    | 88.9    |
| docstring  | 50   | 1.0   | 45        | 0.767  | 0.8067  | 68.9    | 84.4    |
| docstring  | 20   | 1.0   | 45        | 0.7239 | 0.7636  | 64.4    | 77.8    |
| name       | 20   | 0.25  | 125       | 0.5669 | 0.5887  | 52.0    | 60.8    |
| name       | 50   | 0.25  | 125       | 0.5663 | 0.5917  | 52.0    | 60.0    |
| name       | 80   | 0.25  | 125       | 0.5661 | 0.5914  | 52.0    | 60.0    |
| name       | 50   | 0.5   | 125       | 0.5551 | 0.5833  | 49.6    | 60.0    |
| name       | 80   | 0.5   | 125       | 0.5538 | 0.5823  | 49.6    | 60.0    |
| name       | 20   | 0.5   | 125       | 0.5535 | 0.5786  | 49.6    | 60.0    |
| name       | 50   | 0.75  | 125       | 0.5494 | 0.5792  | 48.8    | 59.2    |
| name       | 80   | 0.75  | 125       | 0.5494 | 0.5792  | 48.8    | 59.2    |
| name       | 20   | 0.75  | 125       | 0.547  | 0.5739  | 48.8    | 59.2    |
| name       | 80   | 1.0   | 125       | 0.5321 | 0.5659  | 46.4    | 58.4    |
| name       | 50   | 1.0   | 125       | 0.5268 | 0.5619  | 45.6    | 58.4    |
| name       | 20   | 1.0   | 125       | 0.5162 | 0.55    | 44.8    | 56.0    |
| name       | 20   | 0.0   | 125       | 0.4812 | 0.5245  | 37.6    | 58.4    |
| name       | 50   | 0.0   | 125       | 0.4669 | 0.5164  | 36.8    | 53.6    |
| name       | 80   | 0.0   | 125       | 0.4662 | 0.5158  | 36.8    | 53.6    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| code       | 80   | 0.25  | 121       | 0.66   | 0.7041  | 57.0    | 71.9    |
| code       | 50   | 0.25  | 121       | 0.6582 | 0.7008  | 57.0    | 71.9    |
| code       | 80   | 0.5   | 121       | 0.6564 | 0.7009  | 57.0    | 71.1    |
| code       | 50   | 0.5   | 121       | 0.6551 | 0.698   | 57.0    | 71.1    |
| code       | 50   | 0.75  | 121       | 0.6534 | 0.6967  | 57.0    | 70.2    |
| code       | 80   | 0.75  | 121       | 0.6524 | 0.6973  | 57.0    | 70.2    |
| code       | 20   | 0.25  | 121       | 0.6407 | 0.6817  | 55.4    | 70.2    |
| code       | 20   | 0.5   | 121       | 0.6389 | 0.6801  | 55.4    | 70.2    |
| code       | 20   | 0.75  | 121       | 0.6338 | 0.6761  | 54.5    | 69.4    |
| code       | 80   | 0.0   | 121       | 0.6285 | 0.6804  | 52.1    | 71.9    |
| code       | 50   | 0.0   | 121       | 0.6238 | 0.6746  | 52.1    | 71.1    |
| code       | 20   | 0.0   | 121       | 0.619  | 0.6653  | 52.1    | 70.2    |
| code       | 80   | 1.0   | 121       | 0.5972 | 0.6557  | 48.8    | 66.1    |
| code       | 50   | 1.0   | 121       | 0.5807 | 0.6415  | 46.3    | 64.5    |
| code       | 20   | 1.0   | 121       | 0.5508 | 0.6121  | 43.8    | 61.2    |
| concept    | 80   | 0.0   | 170       | 0.5444 | 0.5745  | 46.5    | 61.2    |
| concept    | 50   | 0.0   | 170       | 0.5022 | 0.5284  | 42.9    | 57.1    |
| concept    | 80   | 0.25  | 170       | 0.4859 | 0.5287  | 40.0    | 54.1    |
| concept    | 80   | 0.5   | 170       | 0.4671 | 0.5138  | 38.2    | 52.4    |
| concept    | 50   | 0.25  | 170       | 0.4594 | 0.4949  | 38.2    | 51.8    |
| concept    | 50   | 0.5   | 170       | 0.4416 | 0.4809  | 36.5    | 50.0    |
| concept    | 20   | 0.0   | 170       | 0.4228 | 0.4453  | 35.9    | 48.8    |
| concept    | 80   | 0.75  | 170       | 0.4164 | 0.4738  | 33.5    | 42.9    |
| concept    | 20   | 0.25  | 170       | 0.403  | 0.4297  | 34.1    | 45.3    |
| concept    | 50   | 0.75  | 170       | 0.4023 | 0.4497  | 32.9    | 42.4    |
| concept    | 80   | 1.0   | 170       | 0.4011 | 0.4612  | 32.4    | 41.8    |
| concept    | 20   | 0.5   | 170       | 0.395  | 0.4234  | 33.5    | 44.1    |
| concept    | 50   | 1.0   | 170       | 0.3795 | 0.4311  | 31.2    | 40.0    |
| concept    | 20   | 0.75  | 170       | 0.372  | 0.4053  | 31.2    | 40.0    |
| concept    | 20   | 1.0   | 170       | 0.3423 | 0.3814  | 28.8    | 35.3    |
| identifier | 20   | 0.25  | 125       | 0.5669 | 0.5887  | 52.0    | 60.8    |
| identifier | 50   | 0.25  | 125       | 0.5663 | 0.5917  | 52.0    | 60.0    |
| identifier | 80   | 0.25  | 125       | 0.5661 | 0.5914  | 52.0    | 60.0    |
| identifier | 50   | 0.5   | 125       | 0.5551 | 0.5833  | 49.6    | 60.0    |
| identifier | 80   | 0.5   | 125       | 0.5538 | 0.5823  | 49.6    | 60.0    |
| identifier | 20   | 0.5   | 125       | 0.5535 | 0.5786  | 49.6    | 60.0    |
| identifier | 50   | 0.75  | 125       | 0.5494 | 0.5792  | 48.8    | 59.2    |
| identifier | 80   | 0.75  | 125       | 0.5494 | 0.5792  | 48.8    | 59.2    |
| identifier | 20   | 0.75  | 125       | 0.547  | 0.5739  | 48.8    | 59.2    |
| identifier | 80   | 1.0   | 125       | 0.5321 | 0.5659  | 46.4    | 58.4    |
| identifier | 50   | 1.0   | 125       | 0.5268 | 0.5619  | 45.6    | 58.4    |
| identifier | 20   | 1.0   | 125       | 0.5162 | 0.55    | 44.8    | 56.0    |
| identifier | 20   | 0.0   | 125       | 0.4812 | 0.5245  | 37.6    | 58.4    |
| identifier | 50   | 0.0   | 125       | 0.4669 | 0.5164  | 36.8    | 53.6    |
| identifier | 80   | 0.0   | 125       | 0.4662 | 0.5158  | 36.8    | 53.6    |

## Per-kind recommendation

| kind       | pool | blend | n_queries | MRR    | NDCG@10 |
| ---------- | ---- | ----- | --------- | ------ | ------- |
| code       | 80   | 0.25  | 121       | 0.66   | 0.7041  |
| concept    | 80   | 0.0   | 170       | 0.5444 | 0.5745  |
| identifier | 20   | 0.25  | 125       | 0.5669 | 0.5887  |

```toml
[reranker_settings.code]
pool = 80
blend_weight = 0.25

[reranker_settings.concept]
pool = 80
blend_weight = 0.0

[reranker_settings.identifier]
pool = 20
blend_weight = 0.25
```

## Per-slug breakdown (best config)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 35        | 0.7152 | 0.7783  | 57.1    | 85.7    |
| astral-sh__uv      | 100       | 0.4853 | 0.5287  | 42.0    | 49.0    |
| badlogic__pi-mono  | 81        | 0.6331 | 0.6508  | 58.0    | 69.1    |
| django__django     | 130       | 0.4899 | 0.531   | 40.8    | 53.8    |
| rbtr__rbtr         | 70        | 0.6384 | 0.6735  | 57.1    | 70.0    |

## Per-language breakdown (best config)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| css        | 30        | 0.725  | 0.7441  | 66.7    | 76.7    |
| html       | 15        | 0.4741 | 0.537   | 33.3    | 53.3    |
| javascript | 40        | 0.9    | 0.9131  | 85.0    | 95.0    |
| json       | 60        | 0.2047 | 0.2447  | 16.7    | 18.3    |
| markdown   | 71        | 0.4439 | 0.4897  | 35.2    | 52.1    |
| python     | 80        | 0.7632 | 0.8054  | 67.5    | 83.8    |
| rst        | 15        | 0.2994 | 0.3701  | 20.0    | 26.7    |
| rust       | 20        | 0.6017 | 0.6749  | 45.0    | 70.0    |
| toml       | 15        | 0.4    | 0.4175  | 33.3    | 46.7    |
| typescript | 40        | 0.8369 | 0.8646  | 77.5    | 90.0    |
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
| elapsed           | 6689 s |

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

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
| 80   | 0.25  | 661       | 0.6022 | 0.6475  | 50.7    | 67.2    | 79.0     | 21.0   | 7206   | 11965  |
| 50   | 0.25  | 661       | 0.6007 | 0.6453  | 50.7    | 66.7    | 78.5     | 21.5   | 4813   | 8154   |
| 80   | 0.5   | 661       | 0.5982 | 0.6445  | 50.2    | 66.9    | 79.0     | 21.0   | 7206   | 11965  |
| 50   | 0.5   | 661       | 0.5943 | 0.6403  | 50.1    | 66.1    | 78.5     | 21.5   | 4813   | 8154   |
| 80   | 0.75  | 661       | 0.5879 | 0.6365  | 49.0    | 66.4    | 79.0     | 21.0   | 7206   | 11965  |
| 50   | 0.75  | 661       | 0.5796 | 0.6291  | 48.1    | 65.1    | 78.5     | 21.5   | 4813   | 8154   |
| 80   | 0.0   | 661       | 0.5748 | 0.6265  | 46.9    | 66.0    | 79.0     | 21.0   | 7206   | 11965  |
| 20   | 0.25  | 661       | 0.5741 | 0.6132  | 49.0    | 63.2    | 73.5     | 26.5   | 2135   | 3860   |
| 50   | 0.0   | 661       | 0.5734 | 0.6244  | 47.0    | 65.5    | 78.5     | 21.5   | 4813   | 8154   |
| 20   | 0.5   | 661       | 0.5733 | 0.6128  | 48.9    | 63.4    | 73.5     | 26.5   | 2135   | 3860   |
| 20   | 0.75  | 661       | 0.5683 | 0.609   | 48.1    | 63.4    | 73.5     | 26.5   | 2135   | 3860   |
| 20   | 0.0   | 661       | 0.5491 | 0.5939  | 45.7    | 62.8    | 73.5     | 26.5   | 2135   | 3860   |
| 80   | 1.0   | 661       | 0.5471 | 0.6047  | 44.3    | 61.3    | 79.0     | 21.0   | 7206   | 11965  |
| 50   | 1.0   | 661       | 0.539  | 0.5971  | 43.9    | 59.3    | 78.5     | 21.5   | 4813   | 8154   |
| 20   | 1.0   | 661       | 0.5234 | 0.5733  | 43.7    | 57.2    | 73.5     | 26.5   | 2135   | 3860   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 185       | 0.738  | 0.7845  | 62.2    | 83.8    |
| body       | 80   | 0.5   | 185       | 0.736  | 0.7827  | 62.7    | 84.9    |
| body       | 50   | 0.25  | 185       | 0.735  | 0.7782  | 62.7    | 82.7    |
| body       | 50   | 0.5   | 185       | 0.7296 | 0.7738  | 62.7    | 83.8    |
| body       | 80   | 0.75  | 185       | 0.7244 | 0.7732  | 61.6    | 81.1    |
| body       | 50   | 0.75  | 185       | 0.7163 | 0.7633  | 61.1    | 80.0    |
| body       | 20   | 0.25  | 185       | 0.7159 | 0.7558  | 61.6    | 80.0    |
| body       | 20   | 0.5   | 185       | 0.7132 | 0.7535  | 61.6    | 80.5    |
| body       | 20   | 0.75  | 185       | 0.705  | 0.747   | 60.5    | 78.4    |
| body       | 80   | 1.0   | 185       | 0.6794 | 0.7387  | 55.7    | 76.8    |
| body       | 50   | 1.0   | 185       | 0.6708 | 0.7278  | 55.7    | 74.6    |
| body       | 80   | 0.0   | 185       | 0.6587 | 0.724   | 50.8    | 79.5    |
| body       | 20   | 0.0   | 185       | 0.6561 | 0.7104  | 52.4    | 78.9    |
| body       | 50   | 0.0   | 185       | 0.655  | 0.7171  | 51.4    | 78.9    |
| body       | 20   | 1.0   | 185       | 0.6379 | 0.694   | 53.5    | 70.3    |
| concept    | 80   | 0.0   | 185       | 0.5174 | 0.5625  | 41.6    | 60.0    |
| concept    | 50   | 0.0   | 185       | 0.5051 | 0.5466  | 41.1    | 58.4    |
| concept    | 80   | 0.25  | 185       | 0.454  | 0.5138  | 33.5    | 54.6    |
| concept    | 50   | 0.25  | 185       | 0.4431 | 0.4992  | 33.0    | 53.0    |
| concept    | 80   | 0.5   | 185       | 0.4253 | 0.4914  | 30.3    | 49.2    |
| concept    | 20   | 0.0   | 185       | 0.4241 | 0.4516  | 35.7    | 49.2    |
| concept    | 50   | 0.5   | 185       | 0.4134 | 0.476   | 29.7    | 47.6    |
| concept    | 20   | 0.25  | 185       | 0.3859 | 0.4228  | 29.7    | 45.9    |
| concept    | 80   | 0.75  | 185       | 0.3757 | 0.4534  | 23.8    | 47.0    |
| concept    | 20   | 0.5   | 185       | 0.3596 | 0.4027  | 26.5    | 42.7    |
| concept    | 50   | 0.75  | 185       | 0.3546 | 0.4309  | 22.2    | 43.2    |
| concept    | 20   | 0.75  | 185       | 0.3155 | 0.3687  | 21.1    | 38.4    |
| concept    | 80   | 1.0   | 185       | 0.3095 | 0.4007  | 17.8    | 36.8    |
| concept    | 50   | 1.0   | 185       | 0.2867 | 0.3766  | 16.2    | 33.0    |
| concept    | 20   | 1.0   | 185       | 0.2526 | 0.3179  | 16.2    | 28.1    |
| docstring  | 50   | 0.25  | 106       | 0.8119 | 0.8355  | 74.5    | 86.8    |
| docstring  | 80   | 0.25  | 106       | 0.8091 | 0.8329  | 74.5    | 86.8    |
| docstring  | 50   | 0.5   | 106       | 0.8087 | 0.8333  | 73.6    | 86.8    |
| docstring  | 50   | 0.75  | 106       | 0.8077 | 0.8324  | 73.6    | 87.7    |
| docstring  | 80   | 0.75  | 106       | 0.8077 | 0.8324  | 73.6    | 87.7    |
| docstring  | 80   | 0.5   | 106       | 0.8073 | 0.8319  | 73.6    | 86.8    |
| docstring  | 20   | 0.75  | 106       | 0.7959 | 0.8166  | 73.6    | 84.9    |
| docstring  | 20   | 0.25  | 106       | 0.7932 | 0.8147  | 72.6    | 85.8    |
| docstring  | 20   | 0.5   | 106       | 0.7925 | 0.814   | 72.6    | 84.9    |
| docstring  | 80   | 1.0   | 106       | 0.7718 | 0.8049  | 68.9    | 84.9    |
| docstring  | 50   | 0.0   | 106       | 0.766  | 0.8011  | 67.9    | 85.8    |
| docstring  | 80   | 0.0   | 106       | 0.7637 | 0.7989  | 67.9    | 85.8    |
| docstring  | 50   | 1.0   | 106       | 0.75   | 0.7878  | 67.0    | 80.2    |
| docstring  | 20   | 0.0   | 106       | 0.7484 | 0.7809  | 66.0    | 84.0    |
| docstring  | 20   | 1.0   | 106       | 0.7219 | 0.7597  | 64.2    | 78.3    |
| name       | 20   | 1.0   | 185       | 0.5661 | 0.6014  | 49.7    | 61.1    |
| name       | 20   | 0.75  | 185       | 0.554  | 0.5924  | 48.1    | 61.1    |
| name       | 50   | 1.0   | 185       | 0.5386 | 0.5776  | 46.5    | 58.4    |
| name       | 80   | 0.75  | 185       | 0.5378 | 0.5707  | 47.6    | 58.9    |
| name       | 50   | 0.75  | 185       | 0.5372 | 0.5766  | 46.5    | 58.9    |
| name       | 80   | 1.0   | 185       | 0.5235 | 0.5598  | 45.4    | 56.8    |
| name       | 20   | 0.5   | 185       | 0.5216 | 0.5667  | 44.9    | 54.6    |
| name       | 50   | 0.5   | 185       | 0.5169 | 0.5606  | 44.3    | 55.1    |
| name       | 80   | 0.5   | 185       | 0.5136 | 0.552   | 44.3    | 55.1    |
| name       | 50   | 0.25  | 185       | 0.503  | 0.5496  | 42.7    | 53.0    |
| name       | 80   | 0.25  | 185       | 0.4962 | 0.5381  | 42.7    | 51.9    |
| name       | 20   | 0.25  | 185       | 0.4949 | 0.5455  | 42.2    | 50.8    |
| name       | 20   | 0.0   | 185       | 0.4528 | 0.5124  | 37.3    | 48.1    |
| name       | 50   | 0.0   | 185       | 0.4499 | 0.508   | 36.8    | 47.6    |
| name       | 80   | 0.0   | 185       | 0.4399 | 0.4941  | 36.2    | 47.0    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| code       | 80   | 0.5   | 103       | 0.866  | 0.8854  | 80.6    | 93.2    |
| code       | 50   | 0.5   | 103       | 0.8612 | 0.8793  | 80.6    | 92.2    |
| code       | 80   | 0.25  | 103       | 0.8579 | 0.8795  | 78.6    | 93.2    |
| code       | 80   | 0.75  | 103       | 0.8547 | 0.877   | 78.6    | 93.2    |
| code       | 50   | 0.25  | 103       | 0.8531 | 0.8734  | 78.6    | 92.2    |
| code       | 20   | 0.5   | 103       | 0.8523 | 0.8702  | 79.6    | 90.3    |
| code       | 50   | 0.75  | 103       | 0.8515 | 0.8721  | 78.6    | 92.2    |
| code       | 20   | 0.25  | 103       | 0.8498 | 0.8685  | 78.6    | 91.3    |
| code       | 20   | 0.75  | 103       | 0.8466 | 0.8659  | 78.6    | 90.3    |
| code       | 80   | 1.0   | 103       | 0.8337 | 0.8609  | 75.7    | 91.3    |
| code       | 50   | 1.0   | 103       | 0.8163 | 0.8452  | 73.8    | 89.3    |
| code       | 20   | 1.0   | 103       | 0.7917 | 0.8235  | 71.8    | 84.5    |
| code       | 20   | 0.0   | 103       | 0.7265 | 0.7757  | 59.2    | 87.4    |
| code       | 80   | 0.0   | 103       | 0.7056 | 0.7644  | 55.3    | 86.4    |
| code       | 50   | 0.0   | 103       | 0.7008 | 0.7582  | 55.3    | 85.4    |
| concept    | 80   | 0.0   | 202       | 0.5366 | 0.5781  | 44.6    | 60.9    |
| concept    | 50   | 0.0   | 202       | 0.5253 | 0.5636  | 44.1    | 59.4    |
| concept    | 80   | 0.25  | 202       | 0.4676 | 0.5254  | 35.1    | 55.4    |
| concept    | 50   | 0.25  | 202       | 0.4575 | 0.5121  | 34.7    | 54.0    |
| concept    | 20   | 0.0   | 202       | 0.442  | 0.4674  | 38.1    | 50.5    |
| concept    | 80   | 0.5   | 202       | 0.4363 | 0.501   | 31.7    | 50.5    |
| concept    | 50   | 0.5   | 202       | 0.4254 | 0.487   | 31.2    | 49.0    |
| concept    | 20   | 0.25  | 202       | 0.3979 | 0.4342  | 31.2    | 46.5    |
| concept    | 80   | 0.75  | 202       | 0.3854 | 0.4617  | 25.2    | 47.5    |
| concept    | 20   | 0.5   | 202       | 0.373  | 0.415   | 28.2    | 43.6    |
| concept    | 50   | 0.75  | 202       | 0.3656 | 0.4407  | 23.8    | 43.6    |
| concept    | 20   | 0.75  | 202       | 0.3285 | 0.3806  | 22.8    | 39.1    |
| concept    | 80   | 1.0   | 202       | 0.3189 | 0.4092  | 18.8    | 38.1    |
| concept    | 50   | 1.0   | 202       | 0.2972 | 0.3864  | 17.3    | 33.7    |
| concept    | 20   | 1.0   | 202       | 0.2641 | 0.3289  | 17.3    | 29.2    |
| identifier | 80   | 0.75  | 356       | 0.6257 | 0.6661  | 53.9    | 69.4    |
| identifier | 20   | 0.75  | 356       | 0.6239 | 0.6643  | 53.7    | 69.4    |
| identifier | 50   | 0.75  | 356       | 0.6224 | 0.6656  | 53.1    | 69.4    |
| identifier | 50   | 0.5   | 356       | 0.6129 | 0.6582  | 52.0    | 68.3    |
| identifier | 80   | 0.5   | 356       | 0.6126 | 0.6562  | 52.0    | 68.5    |
| identifier | 50   | 0.25  | 356       | 0.6089 | 0.655   | 51.7    | 66.6    |
| identifier | 20   | 0.5   | 356       | 0.6063 | 0.6505  | 51.7    | 66.9    |
| identifier | 80   | 0.25  | 356       | 0.6047 | 0.6497  | 51.4    | 66.3    |
| identifier | 50   | 1.0   | 356       | 0.596  | 0.6449  | 50.3    | 65.2    |
| identifier | 20   | 0.25  | 356       | 0.5942 | 0.6408  | 50.6    | 64.6    |
| identifier | 80   | 1.0   | 356       | 0.5936 | 0.6415  | 49.7    | 65.7    |
| identifier | 20   | 1.0   | 356       | 0.593  | 0.6397  | 50.6    | 65.2    |
| identifier | 50   | 0.0   | 356       | 0.5639 | 0.6201  | 46.3    | 63.2    |
| identifier | 80   | 0.0   | 356       | 0.5585 | 0.614   | 45.8    | 62.9    |
| identifier | 20   | 0.0   | 356       | 0.5585 | 0.613   | 46.1    | 62.6    |

## Per-kind recommendation

| kind       | pool | blend | n_queries | MRR    | NDCG@10 |
| ---------- | ---- | ----- | --------- | ------ | ------- |
| code       | 80   | 0.5   | 103       | 0.866  | 0.8854  |
| concept    | 80   | 0.0   | 202       | 0.5366 | 0.5781  |
| identifier | 80   | 0.75  | 356       | 0.6257 | 0.6661  |

```toml
[reranker_settings.code]
pool = 80
blend_weight = 0.5

[reranker_settings.concept]
pool = 80
blend_weight = 0.0

[reranker_settings.identifier]
pool = 80
blend_weight = 0.75
```

## Per-slug breakdown (best config)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 145       | 0.7148 | 0.7646  | 62.1    | 77.9    |
| astral-sh__uv      | 165       | 0.587  | 0.6398  | 47.9    | 66.7    |
| badlogic__pi-mono  | 110       | 0.6132 | 0.6601  | 51.8    | 66.4    |
| django__django     | 151       | 0.4849 | 0.5154  | 41.7    | 54.3    |
| rbtr__rbtr         | 90        | 0.6322 | 0.6795  | 51.1    | 73.3    |

## Per-language breakdown (best config)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
|            | 45        | 0.3073 | 0.412   | 13.3    | 35.6    |
| bash       | 60        | 0.7878 | 0.8268  | 71.7    | 85.0    |
| css        | 80        | 0.7786 | 0.8134  | 71.2    | 82.5    |
| html       | 15        | 0.3333 | 0.3333  | 33.3    | 33.3    |
| javascript | 60        | 0.8482 | 0.8775  | 76.7    | 93.3    |
| json       | 75        | 0.295  | 0.3432  | 18.7    | 38.7    |
| markdown   | 75        | 0.4423 | 0.4922  | 36.0    | 48.0    |
| python     | 80        | 0.7983 | 0.8239  | 72.5    | 87.5    |
| rst        | 15        | 0.2467 | 0.3165  | 13.3    | 33.3    |
| rust       | 20        | 0.5042 | 0.6021  | 30.0    | 70.0    |
| sql        | 20        | 0.6225 | 0.6802  | 45.0    | 75.0    |
| toml       | 20        | 0.53   | 0.5591  | 45.0    | 60.0    |
| typescript | 60        | 0.7639 | 0.8017  | 66.7    | 83.3    |
| yaml       | 36        | 0.4716 | 0.5253  | 36.1    | 52.8    |

## Recommended config (flat)

```toml
[reranker]
pool = 80
blend_weight = 0.25
```

## Run metadata

| field             | value   |
| ----------------- | ------- |
| queries evaluated | 661     |
| configs evaluated | 15      |
| elapsed           | 10100 s |

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
| anthropics__skills |            | body       | 5         |
| anthropics__skills |            | concept    | 5         |
| anthropics__skills |            | name       | 5         |
| anthropics__skills | bash       | body       | 5         |
| anthropics__skills | bash       | concept    | 5         |
| anthropics__skills | bash       | docstring  | 5         |
| anthropics__skills | bash       | name       | 5         |
| anthropics__skills | css        | body       | 5         |
| anthropics__skills | css        | concept    | 5         |
| anthropics__skills | css        | docstring  | 5         |
| anthropics__skills | css        | name       | 5         |
| anthropics__skills | javascript | body       | 5         |
| anthropics__skills | javascript | concept    | 5         |
| anthropics__skills | javascript | docstring  | 5         |
| anthropics__skills | javascript | name       | 5         |
| anthropics__skills | json       | body       | 5         |
| anthropics__skills | json       | concept    | 5         |
| anthropics__skills | json       | name       | 5         |
| anthropics__skills | markdown   | body       | 5         |
| anthropics__skills | markdown   | concept    | 5         |
| anthropics__skills | markdown   | name       | 5         |
| anthropics__skills | python     | body       | 5         |
| anthropics__skills | python     | concept    | 5         |
| anthropics__skills | python     | docstring  | 5         |
| anthropics__skills | python     | name       | 5         |
| anthropics__skills | typescript | body       | 5         |
| anthropics__skills | typescript | concept    | 5         |
| anthropics__skills | typescript | docstring  | 5         |
| anthropics__skills | typescript | name       | 5         |
| astral-sh__uv      |            | body       | 5         |
| astral-sh__uv      |            | concept    | 5         |
| astral-sh__uv      |            | name       | 5         |
| astral-sh__uv      | bash       | body       | 5         |
| astral-sh__uv      | bash       | concept    | 5         |
| astral-sh__uv      | bash       | docstring  | 5         |
| astral-sh__uv      | bash       | name       | 5         |
| astral-sh__uv      | css        | body       | 5         |
| astral-sh__uv      | css        | concept    | 5         |
| astral-sh__uv      | css        | docstring  | 5         |
| astral-sh__uv      | css        | name       | 5         |
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
| astral-sh__uv      | toml       | docstring  | 5         |
| astral-sh__uv      | toml       | name       | 5         |
| astral-sh__uv      | yaml       | body       | 5         |
| astral-sh__uv      | yaml       | concept    | 5         |
| astral-sh__uv      | yaml       | docstring  | 5         |
| astral-sh__uv      | yaml       | name       | 5         |
| badlogic__pi-mono  | bash       | body       | 5         |
| badlogic__pi-mono  | bash       | concept    | 5         |
| badlogic__pi-mono  | bash       | docstring  | 5         |
| badlogic__pi-mono  | bash       | name       | 5         |
| badlogic__pi-mono  | css        | body       | 5         |
| badlogic__pi-mono  | css        | concept    | 5         |
| badlogic__pi-mono  | css        | docstring  | 5         |
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
| django__django     |            | body       | 5         |
| django__django     |            | concept    | 5         |
| django__django     |            | name       | 5         |
| django__django     | css        | body       | 5         |
| django__django     | css        | concept    | 5         |
| django__django     | css        | docstring  | 5         |
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
| django__django     | yaml       | docstring  | 1         |
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
| rbtr__rbtr         | sql        | body       | 5         |
| rbtr__rbtr         | sql        | concept    | 5         |
| rbtr__rbtr         | sql        | docstring  | 5         |
| rbtr__rbtr         | sql        | name       | 5         |
| rbtr__rbtr         | typescript | body       | 5         |
| rbtr__rbtr         | typescript | concept    | 5         |
| rbtr__rbtr         | typescript | docstring  | 5         |
| rbtr__rbtr         | typescript | name       | 5         |

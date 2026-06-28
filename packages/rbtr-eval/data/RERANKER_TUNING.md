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
| 80   | 0.25  | 436       | 0.5665 | 0.6055  | 49.1    | 61.5    | 72.9     | 27.1   | 7841   | 13197  |
| 50   | 0.25  | 436       | 0.5592 | 0.5961  | 48.6    | 60.8    | 71.3     | 28.7   | 5425   | 8808   |
| 80   | 0.5   | 436       | 0.5589 | 0.5998  | 47.7    | 61.5    | 72.9     | 27.1   | 7841   | 13197  |
| 50   | 0.5   | 436       | 0.5524 | 0.5911  | 47.2    | 61.0    | 71.3     | 28.7   | 5425   | 8808   |
| 80   | 0.75  | 436       | 0.5468 | 0.5903  | 46.6    | 59.4    | 72.9     | 27.1   | 7841   | 13197  |
| 50   | 0.75  | 436       | 0.5419 | 0.5829  | 46.3    | 58.9    | 71.3     | 28.7   | 5425   | 8808   |
| 80   | 0.0   | 436       | 0.5412 | 0.5865  | 44.7    | 61.2    | 72.9     | 27.1   | 7841   | 13197  |
| 20   | 0.5   | 436       | 0.5331 | 0.5656  | 46.3    | 58.3    | 66.7     | 33.3   | 2492   | 4794   |
| 50   | 0.0   | 436       | 0.533  | 0.5764  | 44.3    | 60.3    | 71.3     | 28.7   | 5425   | 8808   |
| 20   | 0.25  | 436       | 0.5329 | 0.5652  | 46.8    | 57.8    | 66.7     | 33.3   | 2492   | 4794   |
| 20   | 0.75  | 436       | 0.523  | 0.558   | 45.0    | 56.9    | 66.7     | 33.3   | 2492   | 4794   |
| 80   | 1.0   | 436       | 0.5161 | 0.5666  | 42.7    | 56.0    | 72.9     | 27.1   | 7841   | 13197  |
| 20   | 0.0   | 436       | 0.5047 | 0.5441  | 42.0    | 58.0    | 66.7     | 33.3   | 2492   | 4794   |
| 50   | 1.0   | 436       | 0.5035 | 0.5531  | 41.5    | 55.0    | 71.3     | 28.7   | 5425   | 8808   |
| 20   | 1.0   | 436       | 0.4774 | 0.5226  | 39.4    | 51.6    | 66.7     | 33.3   | 2492   | 4794   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 126       | 0.7026 | 0.7398  | 62.7    | 74.6    |
| body       | 50   | 0.25  | 126       | 0.7006 | 0.7363  | 62.7    | 74.6    |
| body       | 80   | 0.5   | 126       | 0.6912 | 0.7308  | 61.1    | 73.8    |
| body       | 50   | 0.5   | 126       | 0.6902 | 0.7283  | 61.1    | 73.8    |
| body       | 80   | 0.75  | 126       | 0.6845 | 0.7257  | 60.3    | 73.0    |
| body       | 50   | 0.75  | 126       | 0.6835 | 0.7232  | 60.3    | 73.0    |
| body       | 20   | 0.25  | 126       | 0.6753 | 0.708   | 60.3    | 72.2    |
| body       | 20   | 0.5   | 126       | 0.6675 | 0.7023  | 58.7    | 72.2    |
| body       | 20   | 0.75  | 126       | 0.6577 | 0.6945  | 57.9    | 70.6    |
| body       | 80   | 0.0   | 126       | 0.6443 | 0.6959  | 53.2    | 73.8    |
| body       | 50   | 0.0   | 126       | 0.6399 | 0.6905  | 53.2    | 73.0    |
| body       | 20   | 0.0   | 126       | 0.632  | 0.6755  | 53.2    | 71.4    |
| body       | 80   | 1.0   | 126       | 0.6307 | 0.6851  | 52.4    | 69.0    |
| body       | 50   | 1.0   | 126       | 0.6146 | 0.6706  | 50.8    | 66.7    |
| body       | 20   | 1.0   | 126       | 0.5876 | 0.6403  | 49.2    | 63.5    |
| concept    | 80   | 0.0   | 130       | 0.4569 | 0.4824  | 40.0    | 50.8    |
| concept    | 50   | 0.0   | 130       | 0.4279 | 0.4495  | 37.7    | 47.7    |
| concept    | 80   | 0.25  | 130       | 0.362  | 0.41    | 26.9    | 42.3    |
| concept    | 80   | 0.5   | 130       | 0.3498 | 0.4004  | 25.4    | 42.3    |
| concept    | 20   | 0.0   | 130       | 0.348  | 0.3662  | 30.8    | 39.2    |
| concept    | 50   | 0.25  | 130       | 0.3406 | 0.383   | 25.4    | 40.0    |
| concept    | 50   | 0.5   | 130       | 0.3299 | 0.3747  | 23.8    | 40.8    |
| concept    | 80   | 0.75  | 130       | 0.3141 | 0.3723  | 21.5    | 36.9    |
| concept    | 80   | 1.0   | 130       | 0.2998 | 0.36    | 21.5    | 32.3    |
| concept    | 50   | 0.75  | 130       | 0.2994 | 0.3506  | 20.8    | 35.4    |
| concept    | 20   | 0.5   | 130       | 0.2983 | 0.3287  | 23.1    | 35.4    |
| concept    | 20   | 0.25  | 130       | 0.2977 | 0.3282  | 23.1    | 34.6    |
| concept    | 50   | 1.0   | 130       | 0.2862 | 0.3393  | 20.8    | 32.3    |
| concept    | 20   | 0.75  | 130       | 0.2701 | 0.3072  | 19.2    | 31.5    |
| concept    | 20   | 1.0   | 130       | 0.2531 | 0.2933  | 18.5    | 28.5    |
| docstring  | 80   | 0.25  | 50        | 0.849  | 0.8766  | 78.0    | 90.0    |
| docstring  | 50   | 0.25  | 50        | 0.844  | 0.868   | 78.0    | 90.0    |
| docstring  | 80   | 0.5   | 50        | 0.829  | 0.8618  | 74.0    | 90.0    |
| docstring  | 50   | 0.5   | 50        | 0.825  | 0.8541  | 74.0    | 90.0    |
| docstring  | 80   | 0.75  | 50        | 0.8123 | 0.8489  | 72.0    | 88.0    |
| docstring  | 50   | 0.75  | 50        | 0.8083 | 0.8412  | 72.0    | 88.0    |
| docstring  | 20   | 0.25  | 50        | 0.802  | 0.826   | 74.0    | 86.0    |
| docstring  | 20   | 0.5   | 50        | 0.7933 | 0.82    | 72.0    | 86.0    |
| docstring  | 80   | 1.0   | 50        | 0.7915 | 0.8326  | 70.0    | 86.0    |
| docstring  | 80   | 0.0   | 50        | 0.7864 | 0.8292  | 68.0    | 86.0    |
| docstring  | 20   | 0.75  | 50        | 0.784  | 0.8132  | 70.0    | 86.0    |
| docstring  | 50   | 0.0   | 50        | 0.779  | 0.8189  | 68.0    | 86.0    |
| docstring  | 50   | 1.0   | 50        | 0.7553 | 0.7998  | 66.0    | 84.0    |
| docstring  | 20   | 0.0   | 50        | 0.742  | 0.7812  | 64.0    | 86.0    |
| docstring  | 20   | 1.0   | 50        | 0.7065 | 0.7537  | 60.0    | 78.0    |
| name       | 20   | 0.75  | 130       | 0.5448 | 0.5783  | 48.5    | 57.7    |
| name       | 50   | 0.75  | 130       | 0.5447 | 0.5798  | 48.5    | 57.7    |
| name       | 80   | 0.75  | 130       | 0.5439 | 0.5776  | 48.5    | 57.7    |
| name       | 20   | 0.5   | 130       | 0.5375 | 0.5723  | 47.7    | 56.9    |
| name       | 50   | 0.5   | 130       | 0.5365 | 0.5735  | 46.9    | 57.7    |
| name       | 80   | 0.5   | 130       | 0.5358 | 0.5713  | 46.9    | 57.7    |
| name       | 50   | 0.25  | 130       | 0.5311 | 0.5687  | 46.9    | 56.9    |
| name       | 80   | 0.25  | 130       | 0.5304 | 0.5665  | 46.9    | 56.9    |
| name       | 20   | 0.25  | 130       | 0.5266 | 0.5634  | 46.9    | 56.2    |
| name       | 50   | 1.0   | 130       | 0.5162 | 0.5581  | 43.8    | 55.4    |
| name       | 80   | 1.0   | 130       | 0.5154 | 0.5559  | 43.8    | 55.4    |
| name       | 20   | 1.0   | 130       | 0.5066 | 0.5489  | 43.1    | 53.1    |
| name       | 20   | 0.0   | 130       | 0.4468 | 0.5035  | 33.8    | 53.1    |
| name       | 50   | 0.0   | 130       | 0.4397 | 0.4996  | 33.1    | 50.8    |
| name       | 80   | 0.0   | 130       | 0.4313 | 0.4912  | 32.3    | 50.0    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| code       | 80   | 0.25  | 126       | 0.7026 | 0.7398  | 62.7    | 74.6    |
| code       | 50   | 0.25  | 126       | 0.7006 | 0.7363  | 62.7    | 74.6    |
| code       | 80   | 0.5   | 126       | 0.6912 | 0.7308  | 61.1    | 73.8    |
| code       | 50   | 0.5   | 126       | 0.6902 | 0.7283  | 61.1    | 73.8    |
| code       | 80   | 0.75  | 126       | 0.6845 | 0.7257  | 60.3    | 73.0    |
| code       | 50   | 0.75  | 126       | 0.6835 | 0.7232  | 60.3    | 73.0    |
| code       | 20   | 0.25  | 126       | 0.6753 | 0.708   | 60.3    | 72.2    |
| code       | 20   | 0.5   | 126       | 0.6675 | 0.7023  | 58.7    | 72.2    |
| code       | 20   | 0.75  | 126       | 0.6577 | 0.6945  | 57.9    | 70.6    |
| code       | 80   | 0.0   | 126       | 0.6443 | 0.6959  | 53.2    | 73.8    |
| code       | 50   | 0.0   | 126       | 0.6399 | 0.6905  | 53.2    | 73.0    |
| code       | 20   | 0.0   | 126       | 0.632  | 0.6755  | 53.2    | 71.4    |
| code       | 80   | 1.0   | 126       | 0.6307 | 0.6851  | 52.4    | 69.0    |
| code       | 50   | 1.0   | 126       | 0.6146 | 0.6706  | 50.8    | 66.7    |
| code       | 20   | 1.0   | 126       | 0.5876 | 0.6403  | 49.2    | 63.5    |
| concept    | 80   | 0.0   | 180       | 0.5484 | 0.5787  | 47.8    | 60.6    |
| concept    | 50   | 0.0   | 180       | 0.5254 | 0.5521  | 46.1    | 58.3    |
| concept    | 80   | 0.25  | 180       | 0.4973 | 0.5396  | 41.1    | 55.6    |
| concept    | 80   | 0.5   | 180       | 0.4829 | 0.5286  | 38.9    | 55.6    |
| concept    | 50   | 0.25  | 180       | 0.4804 | 0.5177  | 40.0    | 53.9    |
| concept    | 50   | 0.5   | 180       | 0.4674 | 0.5079  | 37.8    | 54.4    |
| concept    | 20   | 0.0   | 180       | 0.4574 | 0.4815  | 40.0    | 52.2    |
| concept    | 80   | 0.75  | 180       | 0.4525 | 0.5047  | 35.6    | 51.1    |
| concept    | 50   | 0.75  | 180       | 0.4407 | 0.4869  | 35.0    | 50.0    |
| concept    | 20   | 0.25  | 180       | 0.4378 | 0.4665  | 37.2    | 48.9    |
| concept    | 80   | 1.0   | 180       | 0.4364 | 0.4913  | 35.0    | 47.2    |
| concept    | 20   | 0.5   | 180       | 0.4358 | 0.4651  | 36.7    | 49.4    |
| concept    | 50   | 1.0   | 180       | 0.4165 | 0.4673  | 33.3    | 46.7    |
| concept    | 20   | 0.75  | 180       | 0.4128 | 0.4478  | 33.3    | 46.7    |
| concept    | 20   | 1.0   | 180       | 0.3791 | 0.4212  | 30.0    | 42.2    |
| identifier | 20   | 0.75  | 130       | 0.5448 | 0.5783  | 48.5    | 57.7    |
| identifier | 50   | 0.75  | 130       | 0.5447 | 0.5798  | 48.5    | 57.7    |
| identifier | 80   | 0.75  | 130       | 0.5439 | 0.5776  | 48.5    | 57.7    |
| identifier | 20   | 0.5   | 130       | 0.5375 | 0.5723  | 47.7    | 56.9    |
| identifier | 50   | 0.5   | 130       | 0.5365 | 0.5735  | 46.9    | 57.7    |
| identifier | 80   | 0.5   | 130       | 0.5358 | 0.5713  | 46.9    | 57.7    |
| identifier | 50   | 0.25  | 130       | 0.5311 | 0.5687  | 46.9    | 56.9    |
| identifier | 80   | 0.25  | 130       | 0.5304 | 0.5665  | 46.9    | 56.9    |
| identifier | 20   | 0.25  | 130       | 0.5266 | 0.5634  | 46.9    | 56.2    |
| identifier | 50   | 1.0   | 130       | 0.5162 | 0.5581  | 43.8    | 55.4    |
| identifier | 80   | 1.0   | 130       | 0.5154 | 0.5559  | 43.8    | 55.4    |
| identifier | 20   | 1.0   | 130       | 0.5066 | 0.5489  | 43.1    | 53.1    |
| identifier | 20   | 0.0   | 130       | 0.4468 | 0.5035  | 33.8    | 53.1    |
| identifier | 50   | 0.0   | 130       | 0.4397 | 0.4996  | 33.1    | 50.8    |
| identifier | 80   | 0.0   | 130       | 0.4313 | 0.4912  | 32.3    | 50.0    |

## Per-kind recommendation

| kind       | pool | blend | n_queries | MRR    | NDCG@10 |
| ---------- | ---- | ----- | --------- | ------ | ------- |
| code       | 80   | 0.25  | 126       | 0.7026 | 0.7398  |
| concept    | 80   | 0.0   | 180       | 0.5484 | 0.5787  |
| identifier | 20   | 0.75  | 130       | 0.5448 | 0.5783  |

```toml
[reranker_settings.code]
pool = 80
blend_weight = 0.25

[reranker_settings.concept]
pool = 80
blend_weight = 0.0

[reranker_settings.identifier]
pool = 20
blend_weight = 0.75
```

## Per-slug breakdown (best config)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 35        | 0.7667 | 0.8115  | 62.9    | 91.4    |
| astral-sh__uv      | 100       | 0.4636 | 0.5071  | 40.0    | 48.0    |
| badlogic__pi-mono  | 81        | 0.6554 | 0.6883  | 58.0    | 72.8    |
| django__django     | 130       | 0.476  | 0.5133  | 40.0    | 51.5    |
| rbtr__rbtr         | 90        | 0.6537 | 0.6934  | 58.9    | 68.9    |

## Per-language breakdown (best config)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| css        | 30        | 0.7722 | 0.8037  | 70.0    | 80.0    |
| html       | 15        | 0.3245 | 0.4041  | 20.0    | 33.3    |
| javascript | 40        | 0.8827 | 0.9057  | 82.5    | 95.0    |
| json       | 60        | 0.2447 | 0.2875  | 20.0    | 23.3    |
| markdown   | 71        | 0.4391 | 0.4796  | 35.2    | 52.1    |
| python     | 80        | 0.7686 | 0.8045  | 67.5    | 85.0    |
| rst        | 15        | 0.2033 | 0.2492  | 13.3    | 20.0    |
| rust       | 20        | 0.7196 | 0.7757  | 60.0    | 80.0    |
| sql        | 20        | 0.6043 | 0.683   | 50.0    | 60.0    |
| toml       | 15        | 0.4    | 0.4     | 40.0    | 40.0    |
| typescript | 40        | 0.864  | 0.8914  | 80.0    | 92.5    |
| yaml       | 30        | 0.207  | 0.2456  | 13.3    | 26.7    |

## Recommended config (flat)

```toml
[reranker]
pool = 80
blend_weight = 0.25
```

## Run metadata

| field             | value  |
| ----------------- | ------ |
| queries evaluated | 440    |
| configs evaluated | 15     |
| elapsed           | 7129 s |

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
| rbtr__rbtr         | sql        | body       | 5         |
| rbtr__rbtr         | sql        | concept    | 5         |
| rbtr__rbtr         | sql        | docstring  | 5         |
| rbtr__rbtr         | sql        | name       | 5         |
| rbtr__rbtr         | typescript | body       | 5         |
| rbtr__rbtr         | typescript | concept    | 5         |
| rbtr__rbtr         | typescript | docstring  | 5         |
| rbtr__rbtr         | typescript | name       | 5         |

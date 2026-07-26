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
| 80   | 0.25  | 661       | 0.5954 | 0.6423  | 50.1    | 66.0    | 79.0     | 21.0   | 7230   | 12359  |
| 50   | 0.25  | 661       | 0.595  | 0.6417  | 50.1    | 65.8    | 78.8     | 21.2   | 4898   | 8542   |
| 80   | 0.5   | 661       | 0.5927 | 0.6403  | 49.6    | 66.7    | 79.0     | 21.0   | 7230   | 12359  |
| 50   | 0.5   | 661       | 0.5897 | 0.6376  | 49.5    | 66.3    | 78.8     | 21.2   | 4898   | 8542   |
| 80   | 0.75  | 661       | 0.5854 | 0.6345  | 49.0    | 65.2    | 79.0     | 21.0   | 7230   | 12359  |
| 50   | 0.75  | 661       | 0.5803 | 0.6301  | 48.4    | 64.6    | 78.8     | 21.2   | 4898   | 8542   |
| 20   | 0.5   | 661       | 0.5706 | 0.6112  | 48.7    | 63.4    | 73.8     | 26.2   | 2233   | 3978   |
| 50   | 0.0   | 661       | 0.5695 | 0.622   | 46.7    | 64.6    | 78.8     | 21.2   | 4898   | 8542   |
| 20   | 0.75  | 661       | 0.5694 | 0.6104  | 48.4    | 63.1    | 73.8     | 26.2   | 2233   | 3978   |
| 20   | 0.25  | 661       | 0.5692 | 0.6101  | 48.6    | 62.5    | 73.8     | 26.2   | 2233   | 3978   |
| 80   | 0.0   | 661       | 0.5688 | 0.6219  | 46.4    | 64.9    | 79.0     | 21.0   | 7230   | 12359  |
| 80   | 1.0   | 661       | 0.5511 | 0.6079  | 44.8    | 61.0    | 79.0     | 21.0   | 7230   | 12359  |
| 20   | 0.0   | 661       | 0.5467 | 0.5927  | 45.5    | 62.0    | 73.8     | 26.2   | 2233   | 3978   |
| 50   | 1.0   | 661       | 0.5457 | 0.603   | 44.6    | 59.3    | 78.8     | 21.2   | 4898   | 8542   |
| 20   | 1.0   | 661       | 0.5252 | 0.5755  | 43.7    | 57.3    | 73.8     | 26.2   | 2233   | 3978   |

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
| concept    | 80   | 0.0   | 185       | 0.4962 | 0.546   | 40.0    | 56.2    |
| concept    | 50   | 0.0   | 185       | 0.491  | 0.5382  | 40.0    | 55.1    |
| concept    | 80   | 0.25  | 185       | 0.4298 | 0.4951  | 31.4    | 50.3    |
| concept    | 50   | 0.25  | 185       | 0.4229 | 0.4862  | 30.8    | 49.7    |
| concept    | 20   | 0.0   | 185       | 0.4165 | 0.448   | 35.1    | 46.5    |
| concept    | 80   | 0.5   | 185       | 0.4055 | 0.4764  | 28.1    | 48.6    |
| concept    | 50   | 0.5   | 185       | 0.3971 | 0.4662  | 27.6    | 48.1    |
| concept    | 20   | 0.25  | 185       | 0.3682 | 0.4115  | 28.1    | 43.2    |
| concept    | 80   | 0.75  | 185       | 0.3668 | 0.4461  | 23.8    | 42.7    |
| concept    | 50   | 0.75  | 185       | 0.357  | 0.4346  | 23.2    | 41.6    |
| concept    | 20   | 0.5   | 185       | 0.3499 | 0.3973  | 25.9    | 42.7    |
| concept    | 80   | 1.0   | 185       | 0.3241 | 0.4122  | 19.5    | 35.7    |
| concept    | 20   | 0.75  | 185       | 0.3203 | 0.3745  | 22.2    | 37.8    |
| concept    | 50   | 1.0   | 185       | 0.3108 | 0.3977  | 18.9    | 33.0    |
| concept    | 20   | 1.0   | 185       | 0.259  | 0.3258  | 16.2    | 28.6    |
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
| name       | 20   | 1.0   | 185       | 0.5659 | 0.6011  | 49.7    | 61.1    |
| name       | 20   | 0.75  | 185       | 0.5531 | 0.5916  | 48.1    | 60.5    |
| name       | 50   | 1.0   | 185       | 0.5386 | 0.5776  | 46.5    | 58.4    |
| name       | 80   | 0.75  | 185       | 0.5378 | 0.5707  | 47.6    | 58.9    |
| name       | 50   | 0.75  | 185       | 0.5372 | 0.5765  | 46.5    | 58.9    |
| name       | 80   | 1.0   | 185       | 0.5235 | 0.5598  | 45.4    | 56.8    |
| name       | 20   | 0.5   | 185       | 0.5216 | 0.5667  | 44.9    | 54.6    |
| name       | 50   | 0.5   | 185       | 0.517  | 0.5607  | 44.3    | 55.1    |
| name       | 80   | 0.5   | 185       | 0.5136 | 0.552   | 44.3    | 55.1    |
| name       | 50   | 0.25  | 185       | 0.503  | 0.5496  | 42.7    | 53.0    |
| name       | 80   | 0.25  | 185       | 0.4962 | 0.5381  | 42.7    | 51.9    |
| name       | 20   | 0.25  | 185       | 0.495  | 0.5456  | 42.2    | 50.8    |
| name       | 20   | 0.0   | 185       | 0.452  | 0.5118  | 37.3    | 48.1    |
| name       | 50   | 0.0   | 185       | 0.4499 | 0.508   | 36.8    | 47.6    |
| name       | 80   | 0.0   | 185       | 0.4399 | 0.4941  | 36.2    | 47.0    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % | p50 ms |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- | ------ |
| code       | 80   | 0.5   | 103       | 0.866  | 0.8854  | 80.6    | 93.2    | 8170   |
| code       | 50   | 0.5   | 103       | 0.8612 | 0.8793  | 80.6    | 92.2    | 5466   |
| code       | 80   | 0.25  | 103       | 0.8579 | 0.8795  | 78.6    | 93.2    | 8170   |
| code       | 80   | 0.75  | 103       | 0.8547 | 0.877   | 78.6    | 93.2    | 8170   |
| code       | 50   | 0.25  | 103       | 0.8531 | 0.8734  | 78.6    | 92.2    | 5466   |
| code       | 20   | 0.5   | 103       | 0.8523 | 0.8702  | 79.6    | 90.3    | 2455   |
| code       | 50   | 0.75  | 103       | 0.8515 | 0.8721  | 78.6    | 92.2    | 5466   |
| code       | 20   | 0.25  | 103       | 0.8498 | 0.8685  | 78.6    | 91.3    | 2455   |
| code       | 20   | 0.75  | 103       | 0.8466 | 0.8659  | 78.6    | 90.3    | 2455   |
| code       | 80   | 1.0   | 103       | 0.8337 | 0.8609  | 75.7    | 91.3    | 8170   |
| code       | 50   | 1.0   | 103       | 0.8163 | 0.8452  | 73.8    | 89.3    | 5466   |
| code       | 20   | 1.0   | 103       | 0.7917 | 0.8235  | 71.8    | 84.5    | 2455   |
| code       | 20   | 0.0   | 103       | 0.7265 | 0.7757  | 59.2    | 87.4    | 2455   |
| code       | 80   | 0.0   | 103       | 0.7056 | 0.7644  | 55.3    | 86.4    | 8170   |
| code       | 50   | 0.0   | 103       | 0.7008 | 0.7582  | 55.3    | 85.4    | 5466   |
| concept    | 80   | 0.0   | 201       | 0.5111 | 0.5569  | 42.3    | 57.2    | 7363   |
| concept    | 50   | 0.0   | 201       | 0.5063 | 0.5497  | 42.3    | 56.2    | 4934   |
| concept    | 80   | 0.25  | 201       | 0.4409 | 0.5034  | 32.8    | 50.7    | 7363   |
| concept    | 50   | 0.25  | 201       | 0.4346 | 0.4952  | 32.3    | 50.2    | 4934   |
| concept    | 20   | 0.0   | 201       | 0.4281 | 0.4571  | 36.8    | 47.3    | 2352   |
| concept    | 80   | 0.5   | 201       | 0.4136 | 0.4822  | 29.4    | 49.3    | 7363   |
| concept    | 50   | 0.5   | 201       | 0.4059 | 0.4729  | 28.9    | 48.8    | 4934   |
| concept    | 20   | 0.25  | 201       | 0.377  | 0.4185  | 29.4    | 43.3    | 2352   |
| concept    | 80   | 0.75  | 201       | 0.3724 | 0.4499  | 24.9    | 42.8    | 7363   |
| concept    | 50   | 0.75  | 201       | 0.363  | 0.4389  | 24.4    | 41.3    | 4934   |
| concept    | 20   | 0.5   | 201       | 0.3593 | 0.4047  | 27.4    | 42.8    | 2352   |
| concept    | 20   | 0.75  | 201       | 0.3278 | 0.3803  | 23.4    | 37.8    | 2352   |
| concept    | 80   | 1.0   | 201       | 0.3277 | 0.4147  | 19.9    | 36.8    | 7363   |
| concept    | 50   | 1.0   | 201       | 0.3146 | 0.4006  | 19.4    | 33.3    | 4934   |
| concept    | 20   | 1.0   | 201       | 0.2653 | 0.3309  | 16.9    | 29.4    | 2352   |
| identifier | 80   | 0.75  | 357       | 0.6277 | 0.6684  | 54.1    | 69.7    | 6894   |
| identifier | 20   | 0.75  | 357       | 0.6254 | 0.6662  | 53.8    | 69.5    | 2148   |
| identifier | 50   | 0.75  | 357       | 0.6244 | 0.6679  | 53.2    | 69.7    | 4666   |
| identifier | 50   | 0.5   | 357       | 0.6149 | 0.6606  | 52.1    | 68.6    | 4666   |
| identifier | 80   | 0.5   | 357       | 0.6147 | 0.6586  | 52.1    | 68.9    | 6894   |
| identifier | 50   | 0.25  | 357       | 0.6109 | 0.6574  | 51.8    | 66.9    | 4666   |
| identifier | 20   | 0.5   | 357       | 0.6083 | 0.6528  | 51.8    | 67.2    | 2148   |
| identifier | 80   | 0.25  | 357       | 0.6067 | 0.6521  | 51.5    | 66.7    | 6894   |
| identifier | 50   | 1.0   | 357       | 0.5978 | 0.6471  | 50.4    | 65.3    | 4666   |
| identifier | 20   | 0.25  | 357       | 0.5964 | 0.6433  | 50.7    | 65.0    | 2148   |
| identifier | 80   | 1.0   | 357       | 0.5954 | 0.6437  | 49.9    | 65.8    | 6894   |
| identifier | 20   | 1.0   | 357       | 0.5946 | 0.6416  | 50.7    | 65.3    | 2148   |
| identifier | 50   | 0.0   | 357       | 0.5673 | 0.6234  | 46.8    | 63.3    | 4666   |
| identifier | 80   | 0.0   | 357       | 0.5619 | 0.6173  | 46.2    | 63.0    | 6894   |
| identifier | 20   | 0.0   | 357       | 0.5616 | 0.6162  | 46.5    | 63.0    | 2148   |

## Latency-aware selection

MRR rises with pool size, but so does latency (p50 is a function of
pool only). For each kind and overall, the cheapest (smallest-pool)
config whose MRR is within a tolerance of that scope's best, and the
MRR given up for it. Choose a pool per your latency budget; the config
is applied manually.

| kind       | tol % | pool | blend | MRR    | p50 ms | MRR vs best |
| ---------- | ----- | ---- | ----- | ------ | ------ | ----------- |
| code       | 0.5   | 80   | 0.5   | 0.866  | 8170   | 0.0         |
| code       | 1.0   | 50   | 0.5   | 0.8612 | 5466   | 0.0049      |
| code       | 2.0   | 20   | 0.5   | 0.8523 | 2455   | 0.0138      |
| code       | 3.0   | 20   | 0.5   | 0.8523 | 2455   | 0.0138      |
| code       | 5.0   | 20   | 0.5   | 0.8523 | 2455   | 0.0138      |
| concept    | 0.5   | 80   | 0.0   | 0.5111 | 7363   | 0.0         |
| concept    | 1.0   | 50   | 0.0   | 0.5063 | 4934   | 0.0048      |
| concept    | 2.0   | 50   | 0.0   | 0.5063 | 4934   | 0.0048      |
| concept    | 3.0   | 50   | 0.0   | 0.5063 | 4934   | 0.0048      |
| concept    | 5.0   | 50   | 0.0   | 0.5063 | 4934   | 0.0048      |
| identifier | 0.5   | 20   | 0.75  | 0.6254 | 2148   | 0.0022      |
| identifier | 1.0   | 20   | 0.75  | 0.6254 | 2148   | 0.0022      |
| identifier | 2.0   | 20   | 0.75  | 0.6254 | 2148   | 0.0022      |
| identifier | 3.0   | 20   | 0.75  | 0.6254 | 2148   | 0.0022      |
| identifier | 5.0   | 20   | 0.75  | 0.6254 | 2148   | 0.0022      |
| all        | 0.5   | 50   | 0.25  | 0.595  | 4898   | 0.0004      |
| all        | 1.0   | 50   | 0.25  | 0.595  | 4898   | 0.0004      |
| all        | 2.0   | 50   | 0.25  | 0.595  | 4898   | 0.0004      |
| all        | 3.0   | 50   | 0.25  | 0.595  | 4898   | 0.0004      |
| all        | 5.0   | 20   | 0.5   | 0.5706 | 2233   | 0.0249      |

## Per-slug breakdown (highest-MRR config: pool 80, blend 0.25)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 145       | 0.7247 | 0.7756  | 62.8    | 79.3    |
| astral-sh__uv      | 165       | 0.5765 | 0.6289  | 47.3    | 63.0    |
| badlogic__pi-mono  | 110       | 0.6186 | 0.6643  | 52.7    | 67.3    |
| django__django     | 151       | 0.4711 | 0.5033  | 40.4    | 52.3    |
| rbtr__rbtr         | 90        | 0.6022 | 0.6585  | 47.8    | 71.1    |

## Per-language breakdown (highest-MRR config: pool 80, blend 0.25)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
|            | 45        | 0.3991 | 0.4972  | 26.7    | 37.8    |
| bash       | 60        | 0.7587 | 0.8088  | 66.7    | 83.3    |
| css        | 80        | 0.7867 | 0.8174  | 71.2    | 83.8    |
| html       | 15        | 0.3333 | 0.3333  | 33.3    | 33.3    |
| javascript | 60        | 0.8325 | 0.8696  | 73.3    | 91.7    |
| json       | 75        | 0.2929 | 0.3473  | 18.7    | 37.3    |
| markdown   | 75        | 0.3924 | 0.4438  | 32.0    | 41.3    |
| python     | 80        | 0.789  | 0.8171  | 71.2    | 87.5    |
| rst        | 15        | 0.2244 | 0.2832  | 13.3    | 26.7    |
| rust       | 20        | 0.5167 | 0.6096  | 35.0    | 65.0    |
| sql        | 20        | 0.65   | 0.7005  | 50.0    | 80.0    |
| toml       | 20        | 0.475  | 0.4946  | 40.0    | 55.0    |
| typescript | 60        | 0.7545 | 0.7986  | 65.0    | 85.0    |
| yaml       | 36        | 0.4512 | 0.5039  | 33.3    | 50.0    |

## Run metadata

| field             | value   |
| ----------------- | ------- |
| queries evaluated | 661     |
| configs evaluated | 15      |
| elapsed           | 10325 s |

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

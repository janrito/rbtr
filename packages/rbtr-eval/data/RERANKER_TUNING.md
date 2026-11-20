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
| 80   | 0.25  | 686       | 0.6163 | 0.6623  | 51.9    | 69.2    | 80.6     | 19.4   | 9810   | 17632  |
| 80   | 0.5   | 686       | 0.6081 | 0.6557  | 51.2    | 67.5    | 80.6     | 19.4   | 9810   | 17632  |
| 50   | 0.25  | 686       | 0.6034 | 0.645   | 51.2    | 67.8    | 77.4     | 22.6   | 6703   | 12563  |
| 50   | 0.5   | 686       | 0.5976 | 0.6404  | 50.7    | 66.5    | 77.4     | 22.6   | 6703   | 12563  |
| 80   | 0.0   | 686       | 0.5812 | 0.6358  | 46.4    | 67.6    | 80.6     | 19.4   | 9810   | 17632  |
| 80   | 0.75  | 686       | 0.5718 | 0.6278  | 46.4    | 64.6    | 80.6     | 19.4   | 9810   | 17632  |
| 50   | 0.0   | 686       | 0.5667 | 0.6173  | 45.2    | 66.5    | 77.4     | 22.6   | 6703   | 12563  |
| 20   | 0.5   | 686       | 0.5644 | 0.5983  | 48.8    | 62.5    | 70.3     | 29.7   | 3320   | 5789   |
| 20   | 0.25  | 686       | 0.5643 | 0.5981  | 48.8    | 63.0    | 70.3     | 29.7   | 3320   | 5789   |
| 50   | 0.75  | 686       | 0.5619 | 0.6129  | 46.1    | 63.3    | 77.4     | 22.6   | 6703   | 12563  |
| 20   | 0.75  | 686       | 0.5387 | 0.5787  | 45.0    | 60.9    | 70.3     | 29.7   | 3320   | 5789   |
| 80   | 1.0   | 686       | 0.534  | 0.5982  | 42.3    | 59.9    | 80.6     | 19.4   | 9810   | 17632  |
| 20   | 0.0   | 686       | 0.5274 | 0.5702  | 43.1    | 60.6    | 70.3     | 29.7   | 3320   | 5789   |
| 50   | 1.0   | 686       | 0.5205 | 0.5803  | 41.5    | 58.2    | 77.4     | 22.6   | 6703   | 12563  |
| 20   | 1.0   | 686       | 0.4913 | 0.5418  | 39.5    | 55.2    | 70.3     | 29.7   | 3320   | 5789   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 200       | 0.7382 | 0.7797  | 63.5    | 84.0    |
| body       | 80   | 0.5   | 200       | 0.7365 | 0.7783  | 63.5    | 83.0    |
| body       | 50   | 0.5   | 200       | 0.7266 | 0.7671  | 63.0    | 81.5    |
| body       | 50   | 0.25  | 200       | 0.7262 | 0.767   | 62.5    | 82.5    |
| body       | 80   | 0.75  | 200       | 0.7239 | 0.7682  | 62.5    | 81.0    |
| body       | 50   | 0.75  | 200       | 0.719  | 0.7609  | 62.5    | 80.0    |
| body       | 20   | 0.25  | 200       | 0.7088 | 0.7429  | 62.0    | 80.5    |
| body       | 20   | 0.5   | 200       | 0.7076 | 0.7419  | 62.0    | 79.5    |
| body       | 20   | 0.75  | 200       | 0.699  | 0.7352  | 61.0    | 78.5    |
| body       | 80   | 1.0   | 200       | 0.6838 | 0.7377  | 56.5    | 79.0    |
| body       | 50   | 1.0   | 200       | 0.6716 | 0.7247  | 55.5    | 77.5    |
| body       | 80   | 0.0   | 200       | 0.6442 | 0.7089  | 48.0    | 78.5    |
| body       | 50   | 0.0   | 200       | 0.6385 | 0.7009  | 48.0    | 77.5    |
| body       | 20   | 0.0   | 200       | 0.6328 | 0.6857  | 49.5    | 76.5    |
| body       | 20   | 1.0   | 200       | 0.6328 | 0.6841  | 52.5    | 72.0    |
| concept    | 80   | 0.0   | 200       | 0.4457 | 0.4962  | 34.5    | 52.5    |
| concept    | 50   | 0.0   | 200       | 0.4007 | 0.441   | 30.5    | 49.5    |
| concept    | 80   | 0.25  | 200       | 0.3921 | 0.4551  | 27.0    | 48.0    |
| concept    | 80   | 0.5   | 200       | 0.3658 | 0.4339  | 25.0    | 42.5    |
| concept    | 50   | 0.25  | 200       | 0.3645 | 0.4132  | 26.0    | 45.0    |
| concept    | 50   | 0.5   | 200       | 0.3465 | 0.3988  | 24.5    | 41.5    |
| concept    | 20   | 0.0   | 200       | 0.2998 | 0.3256  | 24.0    | 36.0    |
| concept    | 20   | 0.25  | 200       | 0.2801 | 0.311   | 20.5    | 35.0    |
| concept    | 80   | 0.75  | 200       | 0.2756 | 0.3645  | 13.0    | 35.0    |
| concept    | 20   | 0.5   | 200       | 0.2741 | 0.3063  | 20.0    | 33.5    |
| concept    | 50   | 0.75  | 200       | 0.2543 | 0.3281  | 12.5    | 33.5    |
| concept    | 20   | 0.75  | 200       | 0.2151 | 0.2616  | 11.5    | 29.5    |
| concept    | 80   | 1.0   | 200       | 0.2148 | 0.3157  | 8.5     | 24.0    |
| concept    | 50   | 1.0   | 200       | 0.1963 | 0.2809  | 8.5     | 22.0    |
| concept    | 20   | 1.0   | 200       | 0.1618 | 0.2189  | 7.0     | 19.5    |
| docstring  | 50   | 0.25  | 106       | 0.8213 | 0.8453  | 74.5    | 89.6    |
| docstring  | 80   | 0.25  | 106       | 0.8213 | 0.8453  | 74.5    | 89.6    |
| docstring  | 50   | 0.5   | 106       | 0.8115 | 0.8379  | 72.6    | 89.6    |
| docstring  | 80   | 0.5   | 106       | 0.8115 | 0.8379  | 72.6    | 89.6    |
| docstring  | 50   | 0.75  | 106       | 0.7942 | 0.825   | 69.8    | 89.6    |
| docstring  | 80   | 0.75  | 106       | 0.7942 | 0.825   | 69.8    | 89.6    |
| docstring  | 20   | 0.25  | 106       | 0.7886 | 0.8109  | 72.6    | 84.9    |
| docstring  | 20   | 0.5   | 106       | 0.78   | 0.8045  | 70.8    | 84.9    |
| docstring  | 50   | 0.0   | 106       | 0.7776 | 0.8121  | 68.9    | 86.8    |
| docstring  | 80   | 0.0   | 106       | 0.7776 | 0.8121  | 68.9    | 86.8    |
| docstring  | 20   | 0.75  | 106       | 0.7678 | 0.7954  | 68.9    | 84.9    |
| docstring  | 80   | 1.0   | 106       | 0.757  | 0.7959  | 66.0    | 84.9    |
| docstring  | 20   | 0.0   | 106       | 0.7433 | 0.7763  | 67.0    | 79.2    |
| docstring  | 50   | 1.0   | 106       | 0.7365 | 0.7796  | 64.2    | 82.1    |
| docstring  | 20   | 1.0   | 106       | 0.7094 | 0.7501  | 62.3    | 78.3    |
| name       | 80   | 0.5   | 180       | 0.6148 | 0.6589  | 53.9    | 65.0    |
| name       | 80   | 0.25  | 180       | 0.6093 | 0.6543  | 53.3    | 64.4    |
| name       | 50   | 0.5   | 180       | 0.6073 | 0.6516  | 53.3    | 63.9    |
| name       | 50   | 0.25  | 180       | 0.604  | 0.649   | 52.8    | 63.9    |
| name       | 80   | 0.75  | 180       | 0.6011 | 0.6483  | 51.7    | 64.4    |
| name       | 20   | 0.5   | 180       | 0.601  | 0.6416  | 53.3    | 62.8    |
| name       | 50   | 0.75  | 180       | 0.5921 | 0.6398  | 51.1    | 62.2    |
| name       | 80   | 1.0   | 180       | 0.5907 | 0.6404  | 50.0    | 63.9    |
| name       | 20   | 0.25  | 180       | 0.5872 | 0.6309  | 51.7    | 61.7    |
| name       | 50   | 1.0   | 180       | 0.5855 | 0.6351  | 49.4    | 62.8    |
| name       | 20   | 0.75  | 180       | 0.5852 | 0.6297  | 50.6    | 62.2    |
| name       | 20   | 1.0   | 180       | 0.5717 | 0.6196  | 47.8    | 62.8    |
| name       | 50   | 0.0   | 180       | 0.547  | 0.6056  | 44.4    | 61.1    |
| name       | 80   | 0.0   | 180       | 0.5462 | 0.6061  | 44.4    | 61.1    |
| name       | 20   | 0.0   | 180       | 0.536  | 0.5924  | 43.3    | 59.4    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % | p50 ms |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- | ------ |
| code       | 80   | 0.5   | 119       | 0.7882 | 0.8182  | 70.6    | 87.4    | 12018  |
| code       | 50   | 0.5   | 119       | 0.784  | 0.8129  | 70.6    | 86.6    | 7451   |
| code       | 80   | 0.25  | 119       | 0.7823 | 0.8141  | 68.9    | 88.2    | 12018  |
| code       | 50   | 0.25  | 119       | 0.7781 | 0.8088  | 68.9    | 87.4    | 7451   |
| code       | 80   | 0.75  | 119       | 0.7761 | 0.8091  | 68.9    | 86.6    | 12018  |
| code       | 50   | 0.75  | 119       | 0.773  | 0.8047  | 68.9    | 85.7    | 7451   |
| code       | 80   | 1.0   | 119       | 0.7523 | 0.7907  | 66.4    | 84.0    | 12018  |
| code       | 20   | 0.5   | 119       | 0.7489 | 0.7722  | 68.1    | 82.4    | 3671   |
| code       | 20   | 0.25  | 119       | 0.7426 | 0.7677  | 66.4    | 83.2    | 3671   |
| code       | 20   | 0.75  | 119       | 0.7426 | 0.7673  | 67.2    | 82.4    | 3671   |
| code       | 50   | 1.0   | 119       | 0.7384 | 0.778   | 64.7    | 83.2    | 7451   |
| code       | 20   | 1.0   | 119       | 0.6966 | 0.7318  | 61.3    | 77.3    | 3671   |
| code       | 50   | 0.0   | 119       | 0.6629 | 0.7221  | 49.6    | 82.4    | 7451   |
| code       | 80   | 0.0   | 119       | 0.6613 | 0.723   | 48.7    | 83.2    | 12018  |
| code       | 20   | 0.0   | 119       | 0.649  | 0.6975  | 50.4    | 79.8    | 3671   |
| concept    | 80   | 0.0   | 215       | 0.4527 | 0.5018  | 35.3    | 53.0    | 9752   |
| concept    | 50   | 0.0   | 215       | 0.4155 | 0.4551  | 32.1    | 50.7    | 6690   |
| concept    | 80   | 0.25  | 215       | 0.3953 | 0.4577  | 27.4    | 48.4    | 9752   |
| concept    | 50   | 0.25  | 215       | 0.3743 | 0.4233  | 27.0    | 46.0    | 6690   |
| concept    | 80   | 0.5   | 215       | 0.3707 | 0.4378  | 25.6    | 43.3    | 9752   |
| concept    | 50   | 0.5   | 215       | 0.3575 | 0.4099  | 25.6    | 42.8    | 6690   |
| concept    | 20   | 0.0   | 215       | 0.317  | 0.3431  | 25.6    | 37.7    | 3492   |
| concept    | 20   | 0.25  | 215       | 0.2953 | 0.3269  | 21.9    | 36.7    | 3492   |
| concept    | 20   | 0.5   | 215       | 0.2896 | 0.3224  | 21.4    | 35.3    | 3492   |
| concept    | 80   | 0.75  | 215       | 0.2791 | 0.3675  | 13.0    | 36.3    | 9752   |
| concept    | 50   | 0.75  | 215       | 0.2636 | 0.338   | 13.0    | 34.9    | 6690   |
| concept    | 80   | 1.0   | 215       | 0.2285 | 0.3267  | 9.8     | 26.5    | 9752   |
| concept    | 20   | 0.75  | 215       | 0.2263 | 0.2745  | 12.1    | 31.2    | 3492   |
| concept    | 50   | 1.0   | 215       | 0.2125 | 0.2964  | 9.8     | 24.7    | 6690   |
| concept    | 20   | 1.0   | 215       | 0.1795 | 0.237   | 8.4     | 22.3    | 3492   |
| identifier | 80   | 0.25  | 352       | 0.6952 | 0.736   | 61.1    | 75.6    | 9121   |
| identifier | 80   | 0.5   | 352       | 0.6922 | 0.734   | 60.2    | 75.6    | 9121   |
| identifier | 50   | 0.25  | 352       | 0.6842 | 0.725   | 59.9    | 74.4    | 6357   |
| identifier | 80   | 0.75  | 352       | 0.6816 | 0.7255  | 59.1    | 74.4    | 9121   |
| identifier | 50   | 0.5   | 352       | 0.6812 | 0.7228  | 59.4    | 74.1    | 6357   |
| identifier | 50   | 0.75  | 352       | 0.6726 | 0.7159  | 58.5    | 73.0    | 6357   |
| identifier | 20   | 0.5   | 352       | 0.67   | 0.708   | 59.1    | 72.4    | 3049   |
| identifier | 20   | 0.25  | 352       | 0.6683 | 0.7065  | 59.4    | 72.2    | 3049   |
| identifier | 20   | 0.75  | 352       | 0.6605 | 0.7008  | 57.7    | 71.9    | 3049   |
| identifier | 80   | 1.0   | 352       | 0.6467 | 0.6989  | 54.0    | 72.2    | 9121   |
| identifier | 50   | 1.0   | 352       | 0.6349 | 0.6869  | 53.1    | 70.2    | 6357   |
| identifier | 80   | 0.0   | 352       | 0.6327 | 0.6883  | 52.3    | 71.3    | 9121   |
| identifier | 50   | 0.0   | 352       | 0.6264 | 0.681   | 51.7    | 70.7    | 6357   |
| identifier | 20   | 0.0   | 352       | 0.6148 | 0.6659  | 51.4    | 68.2    | 3049   |
| identifier | 20   | 1.0   | 352       | 0.6123 | 0.6637  | 51.1    | 67.9    | 3049   |

## Latency-aware selection

MRR rises with pool size, but so does latency (p50 is a function of
pool only). For each kind and overall, the cheapest (smallest-pool)
config whose MRR is within a tolerance of that scope's best, and the
MRR given up for it. Choose a pool per your latency budget; the config
is applied manually.

| kind       | tol % | pool | blend | MRR    | p50 ms | MRR vs best |
| ---------- | ----- | ---- | ----- | ------ | ------ | ----------- |
| code       | 0.5   | 80   | 0.5   | 0.7882 | 12018  | 0.0         |
| code       | 1.0   | 50   | 0.5   | 0.784  | 7451   | 0.0042      |
| code       | 2.0   | 50   | 0.5   | 0.784  | 7451   | 0.0042      |
| code       | 3.0   | 50   | 0.5   | 0.784  | 7451   | 0.0042      |
| code       | 5.0   | 20   | 0.5   | 0.7489 | 3671   | 0.0393      |
| concept    | 0.5   | 80   | 0.0   | 0.4527 | 9752   | 0.0         |
| concept    | 1.0   | 80   | 0.0   | 0.4527 | 9752   | 0.0         |
| concept    | 2.0   | 80   | 0.0   | 0.4527 | 9752   | 0.0         |
| concept    | 3.0   | 80   | 0.0   | 0.4527 | 9752   | 0.0         |
| concept    | 5.0   | 80   | 0.0   | 0.4527 | 9752   | 0.0         |
| identifier | 0.5   | 80   | 0.25  | 0.6952 | 9121   | 0.0         |
| identifier | 1.0   | 80   | 0.25  | 0.6952 | 9121   | 0.0         |
| identifier | 2.0   | 50   | 0.25  | 0.6842 | 6357   | 0.0109      |
| identifier | 3.0   | 50   | 0.25  | 0.6842 | 6357   | 0.0109      |
| identifier | 5.0   | 20   | 0.5   | 0.67   | 3049   | 0.0252      |
| all        | 0.5   | 80   | 0.25  | 0.6163 | 9810   | 0.0         |
| all        | 1.0   | 80   | 0.25  | 0.6163 | 9810   | 0.0         |
| all        | 2.0   | 80   | 0.25  | 0.6163 | 9810   | 0.0         |
| all        | 3.0   | 50   | 0.25  | 0.6034 | 6703   | 0.0129      |
| all        | 5.0   | 50   | 0.25  | 0.6034 | 6703   | 0.0129      |

## Per-slug breakdown (highest-MRR config: pool 80, blend 0.25)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 140       | 0.6947 | 0.7478  | 57.9    | 77.1    |
| astral-sh__uv      | 175       | 0.6242 | 0.6599  | 54.9    | 67.4    |
| badlogic__pi-mono  | 110       | 0.6361 | 0.6831  | 55.5    | 68.2    |
| django__django     | 161       | 0.5447 | 0.5894  | 44.7    | 64.6    |
| rbtr__rbtr         | 100       | 0.5863 | 0.6411  | 46.0    | 70.0    |

## Per-language breakdown (highest-MRR config: pool 80, blend 0.25)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| bash       | 85        | 0.7321 | 0.77    | 67.1    | 76.5    |
| css        | 80        | 0.7826 | 0.8268  | 67.5    | 88.8    |
| html       | 15        | 0.354  | 0.3793  | 33.3    | 33.3    |
| javascript | 60        | 0.7653 | 0.8105  | 65.0    | 86.7    |
| json       | 75        | 0.3427 | 0.3896  | 24.0    | 44.0    |
| markdown   | 75        | 0.3881 | 0.4412  | 30.7    | 41.3    |
| plaintext  | 30        | 0.4684 | 0.5505  | 26.7    | 66.7    |
| python     | 80        | 0.7831 | 0.8156  | 70.0    | 87.5    |
| rst        | 30        | 0.5118 | 0.5472  | 46.7    | 50.0    |
| rust       | 20        | 0.6146 | 0.6834  | 50.0    | 70.0    |
| sql        | 20        | 0.6558 | 0.7164  | 50.0    | 75.0    |
| toml       | 20        | 0.7    | 0.7131  | 65.0    | 75.0    |
| typescript | 60        | 0.7218 | 0.7689  | 61.7    | 80.0    |
| yaml       | 36        | 0.476  | 0.5427  | 33.3    | 58.3    |

## Run metadata

| field             | value   |
| ----------------- | ------- |
| queries evaluated | 686     |
| configs evaluated | 15      |
| elapsed           | 14563 s |

## Sample distribution

| slug               | language   | provenance | n_queries |
| ------------------ | ---------- | ---------- | --------- |
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
| anthropics__skills | plaintext  | body       | 5         |
| anthropics__skills | plaintext  | concept    | 5         |
| anthropics__skills | python     | body       | 5         |
| anthropics__skills | python     | concept    | 5         |
| anthropics__skills | python     | docstring  | 5         |
| anthropics__skills | python     | name       | 5         |
| anthropics__skills | typescript | body       | 5         |
| anthropics__skills | typescript | concept    | 5         |
| anthropics__skills | typescript | docstring  | 5         |
| anthropics__skills | typescript | name       | 5         |
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
| astral-sh__uv      | plaintext  | body       | 5         |
| astral-sh__uv      | plaintext  | concept    | 5         |
| astral-sh__uv      | python     | body       | 5         |
| astral-sh__uv      | python     | concept    | 5         |
| astral-sh__uv      | python     | docstring  | 5         |
| astral-sh__uv      | python     | name       | 5         |
| astral-sh__uv      | rst        | body       | 5         |
| astral-sh__uv      | rst        | concept    | 5         |
| astral-sh__uv      | rst        | name       | 5         |
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
| django__django     | bash       | body       | 5         |
| django__django     | bash       | concept    | 5         |
| django__django     | bash       | name       | 5         |
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
| django__django     | plaintext  | body       | 5         |
| django__django     | plaintext  | concept    | 5         |
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
| rbtr__rbtr         | bash       | body       | 5         |
| rbtr__rbtr         | bash       | concept    | 5         |
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

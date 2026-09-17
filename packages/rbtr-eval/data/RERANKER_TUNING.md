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
| 80   | 0.25  | 686       | 0.6041 | 0.6503  | 50.7    | 67.8    | 79.4     | 20.6   | 12003  | 21972  |
| 50   | 0.25  | 686       | 0.5938 | 0.6383  | 49.9    | 66.8    | 77.7     | 22.3   | 7555   | 13581  |
| 80   | 0.5   | 686       | 0.5918 | 0.6408  | 49.1    | 66.3    | 79.4     | 20.6   | 12003  | 21972  |
| 50   | 0.5   | 686       | 0.583  | 0.6299  | 48.5    | 65.2    | 77.7     | 22.3   | 7555   | 13581  |
| 80   | 0.0   | 686       | 0.579  | 0.6314  | 46.5    | 66.5    | 79.4     | 20.6   | 12003  | 21972  |
| 80   | 0.75  | 686       | 0.577  | 0.6292  | 47.5    | 64.0    | 79.4     | 20.6   | 12003  | 21972  |
| 50   | 0.0   | 686       | 0.5693 | 0.6199  | 45.8    | 65.3    | 77.7     | 22.3   | 7555   | 13581  |
| 50   | 0.75  | 686       | 0.5668 | 0.6173  | 46.8    | 62.7    | 77.7     | 22.3   | 7555   | 13581  |
| 20   | 0.25  | 686       | 0.5616 | 0.5989  | 48.0    | 62.7    | 71.4     | 28.6   | 4319   | 6461   |
| 20   | 0.5   | 686       | 0.5574 | 0.5956  | 47.4    | 61.7    | 71.4     | 28.6   | 4319   | 6461   |
| 80   | 1.0   | 686       | 0.5465 | 0.6053  | 44.2    | 60.6    | 79.4     | 20.6   | 12003  | 21972  |
| 20   | 0.0   | 686       | 0.5455 | 0.5869  | 45.2    | 62.4    | 71.4     | 28.6   | 4319   | 6461   |
| 20   | 0.75  | 686       | 0.5437 | 0.585   | 45.6    | 60.6    | 71.4     | 28.6   | 4319   | 6461   |
| 50   | 1.0   | 686       | 0.5321 | 0.59    | 43.1    | 58.3    | 77.7     | 22.3   | 7555   | 13581  |
| 20   | 1.0   | 686       | 0.5005 | 0.551   | 41.0    | 55.2    | 71.4     | 28.6   | 4319   | 6461   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 200       | 0.7154 | 0.7577  | 60.5    | 82.0    |
| body       | 50   | 0.25  | 200       | 0.7136 | 0.7551  | 60.5    | 82.0    |
| body       | 80   | 0.5   | 200       | 0.7102 | 0.7537  | 60.0    | 81.5    |
| body       | 50   | 0.5   | 200       | 0.7077 | 0.7505  | 60.0    | 81.0    |
| body       | 20   | 0.25  | 200       | 0.6997 | 0.7373  | 60.0    | 80.0    |
| body       | 80   | 0.75  | 200       | 0.6986 | 0.7447  | 58.5    | 80.5    |
| body       | 50   | 0.75  | 200       | 0.6968 | 0.7421  | 58.5    | 80.0    |
| body       | 20   | 0.5   | 200       | 0.6917 | 0.7311  | 59.0    | 78.5    |
| body       | 20   | 0.75  | 200       | 0.682  | 0.7237  | 57.5    | 78.5    |
| body       | 80   | 1.0   | 200       | 0.6591 | 0.7145  | 53.0    | 78.0    |
| body       | 50   | 1.0   | 200       | 0.6447 | 0.7019  | 52.0    | 75.0    |
| body       | 20   | 0.0   | 200       | 0.6408 | 0.6927  | 51.0    | 75.5    |
| body       | 50   | 0.0   | 200       | 0.6316 | 0.693   | 48.0    | 75.5    |
| body       | 80   | 0.0   | 200       | 0.6273 | 0.6909  | 47.0    | 75.5    |
| body       | 20   | 1.0   | 200       | 0.6151 | 0.6718  | 49.5    | 71.5    |
| concept    | 80   | 0.0   | 200       | 0.4572 | 0.5025  | 36.5    | 51.5    |
| concept    | 50   | 0.0   | 200       | 0.4259 | 0.4682  | 33.5    | 48.5    |
| concept    | 80   | 0.25  | 200       | 0.3792 | 0.4428  | 26.5    | 45.0    |
| concept    | 20   | 0.0   | 200       | 0.3603 | 0.386   | 30.0    | 41.0    |
| concept    | 50   | 0.25  | 200       | 0.3551 | 0.414   | 24.5    | 42.5    |
| concept    | 80   | 0.5   | 200       | 0.3456 | 0.4168  | 22.0    | 41.0    |
| concept    | 50   | 0.5   | 200       | 0.3286 | 0.3934  | 21.0    | 38.5    |
| concept    | 80   | 0.75  | 200       | 0.3204 | 0.3969  | 20.0    | 35.0    |
| concept    | 50   | 0.75  | 200       | 0.3006 | 0.3715  | 18.5    | 33.5    |
| concept    | 20   | 0.25  | 200       | 0.2902 | 0.3326  | 20.5    | 34.5    |
| concept    | 80   | 1.0   | 200       | 0.2816 | 0.3658  | 16.5    | 30.0    |
| concept    | 20   | 0.5   | 200       | 0.2758 | 0.3215  | 18.5    | 32.0    |
| concept    | 50   | 1.0   | 200       | 0.2607 | 0.3391  | 15.5    | 27.0    |
| concept    | 20   | 0.75  | 200       | 0.2531 | 0.3039  | 16.0    | 30.0    |
| concept    | 20   | 1.0   | 200       | 0.2101 | 0.2689  | 13.0    | 22.5    |
| docstring  | 80   | 0.25  | 106       | 0.8269 | 0.8514  | 75.5    | 89.6    |
| docstring  | 50   | 0.25  | 106       | 0.8175 | 0.842   | 74.5    | 88.7    |
| docstring  | 80   | 0.5   | 106       | 0.8149 | 0.8423  | 73.6    | 88.7    |
| docstring  | 80   | 0.75  | 106       | 0.8132 | 0.841   | 73.6    | 88.7    |
| docstring  | 50   | 0.5   | 106       | 0.8055 | 0.8329  | 72.6    | 87.7    |
| docstring  | 50   | 0.75  | 106       | 0.8014 | 0.8296  | 72.6    | 86.8    |
| docstring  | 20   | 0.25  | 106       | 0.7909 | 0.8128  | 72.6    | 84.9    |
| docstring  | 80   | 0.0   | 106       | 0.7883 | 0.8226  | 69.8    | 88.7    |
| docstring  | 20   | 0.75  | 106       | 0.7858 | 0.8086  | 72.6    | 84.0    |
| docstring  | 20   | 0.5   | 106       | 0.7825 | 0.8062  | 71.7    | 84.0    |
| docstring  | 80   | 1.0   | 106       | 0.7788 | 0.8137  | 70.8    | 84.0    |
| docstring  | 50   | 0.0   | 106       | 0.7784 | 0.8128  | 68.9    | 87.7    |
| docstring  | 50   | 1.0   | 106       | 0.7595 | 0.7964  | 68.9    | 81.1    |
| docstring  | 20   | 0.0   | 106       | 0.7545 | 0.7851  | 67.9    | 83.0    |
| docstring  | 20   | 1.0   | 106       | 0.7278 | 0.7633  | 66.0    | 78.3    |
| name       | 80   | 0.5   | 180       | 0.6026 | 0.6455  | 52.8    | 64.4    |
| name       | 80   | 0.25  | 180       | 0.5991 | 0.6429  | 52.2    | 64.4    |
| name       | 50   | 0.5   | 180       | 0.596  | 0.6391  | 52.2    | 63.9    |
| name       | 50   | 0.25  | 180       | 0.5941 | 0.6378  | 51.7    | 63.9    |
| name       | 20   | 0.5   | 180       | 0.5883 | 0.6256  | 52.2    | 62.8    |
| name       | 80   | 0.75  | 180       | 0.588  | 0.6342  | 50.6    | 63.3    |
| name       | 50   | 0.75  | 180       | 0.5801 | 0.6268  | 50.0    | 61.7    |
| name       | 80   | 1.0   | 180       | 0.5789 | 0.6271  | 49.4    | 61.7    |
| name       | 20   | 0.25  | 180       | 0.5748 | 0.6152  | 50.6    | 61.7    |
| name       | 50   | 1.0   | 180       | 0.5747 | 0.6228  | 48.9    | 61.1    |
| name       | 20   | 0.75  | 180       | 0.5702 | 0.6116  | 49.4    | 61.1    |
| name       | 20   | 1.0   | 180       | 0.5618 | 0.6053  | 47.8    | 60.0    |
| name       | 80   | 0.0   | 180       | 0.5375 | 0.5959  | 43.3    | 60.0    |
| name       | 50   | 0.0   | 180       | 0.5363 | 0.5938  | 43.3    | 59.4    |
| name       | 20   | 0.0   | 180       | 0.5225 | 0.5757  | 42.2    | 59.4    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % | p50 ms |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- | ------ |
| code       | 80   | 0.5   | 119       | 0.7677 | 0.8006  | 68.1    | 84.9    | 13697  |
| code       | 80   | 0.25  | 119       | 0.7674 | 0.8006  | 67.2    | 85.7    | 13697  |
| code       | 50   | 0.5   | 119       | 0.7635 | 0.7952  | 68.1    | 84.0    | 8463   |
| code       | 50   | 0.25  | 119       | 0.7632 | 0.7953  | 67.2    | 84.9    | 8463   |
| code       | 80   | 0.75  | 119       | 0.7597 | 0.7945  | 67.2    | 84.0    | 13697  |
| code       | 50   | 0.75  | 119       | 0.7567 | 0.7901  | 67.2    | 83.2    | 8463   |
| code       | 20   | 0.25  | 119       | 0.7406 | 0.77    | 65.5    | 82.4    | 4787   |
| code       | 20   | 0.5   | 119       | 0.7401 | 0.7694  | 66.4    | 80.7    | 4787   |
| code       | 20   | 0.75  | 119       | 0.7388 | 0.7682  | 66.4    | 80.7    | 4787   |
| code       | 80   | 1.0   | 119       | 0.7388 | 0.7783  | 64.7    | 83.2    | 13697  |
| code       | 50   | 1.0   | 119       | 0.7214 | 0.7628  | 63.0    | 79.8    | 8463   |
| code       | 20   | 1.0   | 119       | 0.6983 | 0.7367  | 61.3    | 76.5    | 4787   |
| code       | 20   | 0.0   | 119       | 0.6578 | 0.708   | 51.3    | 79.8    | 4787   |
| code       | 50   | 0.0   | 119       | 0.6472 | 0.708   | 47.9    | 80.7    | 8463   |
| code       | 80   | 0.0   | 119       | 0.6456 | 0.7089  | 47.1    | 81.5    | 13697  |
| concept    | 80   | 0.0   | 215       | 0.4657 | 0.5094  | 37.7    | 52.1    | 12576  |
| concept    | 50   | 0.0   | 215       | 0.4366 | 0.4774  | 34.9    | 49.3    | 7708   |
| concept    | 80   | 0.25  | 215       | 0.3897 | 0.4512  | 27.9    | 46.0    | 12576  |
| concept    | 20   | 0.0   | 215       | 0.3756 | 0.401   | 31.6    | 42.3    | 4424   |
| concept    | 50   | 0.25  | 215       | 0.3673 | 0.4244  | 26.0    | 43.7    | 7708   |
| concept    | 80   | 0.5   | 215       | 0.3584 | 0.427   | 23.7    | 42.3    | 12576  |
| concept    | 50   | 0.5   | 215       | 0.3426 | 0.4052  | 22.8    | 40.0    | 7708   |
| concept    | 80   | 0.75  | 215       | 0.3295 | 0.4043  | 20.9    | 36.7    | 12576  |
| concept    | 50   | 0.75  | 215       | 0.3108 | 0.3804  | 19.5    | 34.9    | 7708   |
| concept    | 20   | 0.25  | 215       | 0.3069 | 0.3487  | 22.3    | 36.3    | 4424   |
| concept    | 80   | 1.0   | 215       | 0.2958 | 0.3772  | 18.1    | 32.1    | 12576  |
| concept    | 20   | 0.5   | 215       | 0.2935 | 0.3383  | 20.5    | 34.0    | 4424   |
| concept    | 50   | 1.0   | 215       | 0.2759 | 0.352   | 17.2    | 28.8    | 7708   |
| concept    | 20   | 0.75  | 215       | 0.2666 | 0.3175  | 17.2    | 31.6    | 4424   |
| concept    | 20   | 1.0   | 215       | 0.2287 | 0.2865  | 14.9    | 24.7    | 4424   |
| identifier | 80   | 0.25  | 352       | 0.6798 | 0.721   | 59.1    | 75.0    | 11007  |
| identifier | 80   | 0.5   | 352       | 0.6749 | 0.7174  | 58.2    | 74.7    | 11007  |
| identifier | 50   | 0.25  | 352       | 0.6748 | 0.7159  | 58.5    | 74.7    | 7229   |
| identifier | 50   | 0.5   | 352       | 0.6687 | 0.7113  | 57.7    | 74.1    | 7229   |
| identifier | 80   | 0.75  | 352       | 0.6664 | 0.7106  | 57.1    | 73.9    | 11007  |
| identifier | 50   | 0.75  | 352       | 0.659  | 0.7036  | 56.5    | 72.7    | 7229   |
| identifier | 20   | 0.5   | 352       | 0.6567 | 0.6941  | 57.4    | 72.2    | 3994   |
| identifier | 20   | 0.25  | 352       | 0.6567 | 0.6939  | 57.7    | 72.2    | 3994   |
| identifier | 20   | 0.75  | 352       | 0.6469 | 0.6865  | 56.0    | 71.6    | 3994   |
| identifier | 80   | 1.0   | 352       | 0.6346 | 0.6861  | 53.1    | 70.5    | 11007  |
| identifier | 80   | 0.0   | 352       | 0.6257 | 0.6797  | 51.7    | 70.2    | 11007  |
| identifier | 50   | 1.0   | 352       | 0.6246 | 0.6769  | 52.3    | 69.0    | 7229   |
| identifier | 50   | 0.0   | 352       | 0.624  | 0.6771  | 51.7    | 69.9    | 7229   |
| identifier | 20   | 0.0   | 352       | 0.6114 | 0.6594  | 51.4    | 68.8    | 3994   |
| identifier | 20   | 1.0   | 352       | 0.5996 | 0.6498  | 50.0    | 66.8    | 3994   |

## Latency-aware selection

MRR rises with pool size, but so does latency (p50 is a function of
pool only). For each kind and overall, the cheapest (smallest-pool)
config whose MRR is within a tolerance of that scope's best, and the
MRR given up for it. Choose a pool per your latency budget; the config
is applied manually.

| kind       | tol % | pool | blend | MRR    | p50 ms | MRR vs best |
| ---------- | ----- | ---- | ----- | ------ | ------ | ----------- |
| code       | 0.5   | 80   | 0.5   | 0.7677 | 13697  | 0.0         |
| code       | 1.0   | 50   | 0.5   | 0.7635 | 8463   | 0.0042      |
| code       | 2.0   | 50   | 0.5   | 0.7635 | 8463   | 0.0042      |
| code       | 3.0   | 50   | 0.5   | 0.7635 | 8463   | 0.0042      |
| code       | 5.0   | 20   | 0.25  | 0.7406 | 4787   | 0.0271      |
| concept    | 0.5   | 80   | 0.0   | 0.4657 | 12576  | 0.0         |
| concept    | 1.0   | 80   | 0.0   | 0.4657 | 12576  | 0.0         |
| concept    | 2.0   | 80   | 0.0   | 0.4657 | 12576  | 0.0         |
| concept    | 3.0   | 80   | 0.0   | 0.4657 | 12576  | 0.0         |
| concept    | 5.0   | 80   | 0.0   | 0.4657 | 12576  | 0.0         |
| identifier | 0.5   | 80   | 0.25  | 0.6798 | 11007  | 0.0         |
| identifier | 1.0   | 50   | 0.25  | 0.6748 | 7229   | 0.005       |
| identifier | 2.0   | 50   | 0.25  | 0.6748 | 7229   | 0.005       |
| identifier | 3.0   | 50   | 0.25  | 0.6748 | 7229   | 0.005       |
| identifier | 5.0   | 20   | 0.5   | 0.6567 | 3994   | 0.0231      |
| all        | 0.5   | 80   | 0.25  | 0.6041 | 12003  | 0.0         |
| all        | 1.0   | 80   | 0.25  | 0.6041 | 12003  | 0.0         |
| all        | 2.0   | 50   | 0.25  | 0.5938 | 7555   | 0.0103      |
| all        | 3.0   | 50   | 0.25  | 0.5938 | 7555   | 0.0103      |
| all        | 5.0   | 50   | 0.25  | 0.5938 | 7555   | 0.0103      |

## Per-slug breakdown (highest-MRR config: pool 80, blend 0.25)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 140       | 0.6937 | 0.7439  | 58.6    | 77.9    |
| astral-sh__uv      | 175       | 0.6177 | 0.655   | 53.7    | 67.4    |
| badlogic__pi-mono  | 110       | 0.6038 | 0.6554  | 50.0    | 67.3    |
| django__django     | 161       | 0.5244 | 0.566   | 44.1    | 59.6    |
| rbtr__rbtr         | 100       | 0.5834 | 0.6409  | 46.0    | 68.0    |

## Per-language breakdown (highest-MRR config: pool 80, blend 0.25)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| bash       | 85        | 0.6905 | 0.7335  | 60.0    | 74.1    |
| css        | 80        | 0.7947 | 0.8394  | 68.8    | 91.2    |
| html       | 15        | 0.3651 | 0.403   | 33.3    | 33.3    |
| javascript | 60        | 0.7496 | 0.795   | 63.3    | 86.7    |
| json       | 75        | 0.3281 | 0.3806  | 22.7    | 40.0    |
| markdown   | 75        | 0.3909 | 0.4449  | 29.3    | 46.7    |
| plaintext  | 30        | 0.492  | 0.5754  | 30.0    | 63.3    |
| python     | 80        | 0.7699 | 0.8073  | 70.0    | 81.2    |
| rst        | 30        | 0.3881 | 0.4132  | 36.7    | 36.7    |
| rust       | 20        | 0.5672 | 0.6346  | 45.0    | 65.0    |
| sql        | 20        | 0.6542 | 0.7149  | 50.0    | 75.0    |
| toml       | 20        | 0.625  | 0.6446  | 55.0    | 70.0    |
| typescript | 60        | 0.7078 | 0.7469  | 61.7    | 78.3    |
| yaml       | 36        | 0.5659 | 0.6104  | 47.2    | 63.9    |

## Run metadata

| field             | value   |
| ----------------- | ------- |
| queries evaluated | 686     |
| configs evaluated | 15      |
| elapsed           | 17379 s |

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

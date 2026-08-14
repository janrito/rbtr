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
| 80   | 0.25  | 686       | 0.6041 | 0.6485  | 51.2    | 67.3    | 78.7     | 21.3   | 9266   | 16096  |
| 50   | 0.25  | 686       | 0.5959 | 0.6378  | 50.6    | 66.8    | 76.8     | 23.2   | 5922   | 10772  |
| 80   | 0.5   | 686       | 0.5952 | 0.6415  | 50.1    | 66.3    | 78.7     | 21.3   | 9266   | 16096  |
| 50   | 0.5   | 686       | 0.5876 | 0.6313  | 49.7    | 65.5    | 76.8     | 23.2   | 5922   | 10772  |
| 80   | 0.75  | 686       | 0.5774 | 0.6276  | 48.0    | 64.3    | 78.7     | 21.3   | 9266   | 16096  |
| 80   | 0.0   | 686       | 0.5687 | 0.6219  | 45.2    | 66.5    | 78.7     | 21.3   | 9266   | 16096  |
| 50   | 0.75  | 686       | 0.568  | 0.616   | 47.4    | 62.8    | 76.8     | 23.2   | 5922   | 10772  |
| 20   | 0.25  | 686       | 0.5615 | 0.5964  | 48.4    | 62.4    | 70.4     | 29.6   | 2693   | 4576   |
| 50   | 0.0   | 686       | 0.5608 | 0.6113  | 44.9    | 65.2    | 76.8     | 23.2   | 5922   | 10772  |
| 20   | 0.5   | 686       | 0.559  | 0.5944  | 48.1    | 61.8    | 70.4     | 29.6   | 2693   | 4576   |
| 20   | 0.75  | 686       | 0.546  | 0.5844  | 46.4    | 61.1    | 70.4     | 29.6   | 2693   | 4576   |
| 80   | 1.0   | 686       | 0.5415 | 0.5997  | 43.7    | 60.8    | 78.7     | 21.3   | 9266   | 16096  |
| 50   | 1.0   | 686       | 0.5302 | 0.5864  | 43.1    | 58.9    | 76.8     | 23.2   | 5922   | 10772  |
| 20   | 0.0   | 686       | 0.5292 | 0.572   | 43.1    | 61.1    | 70.4     | 29.6   | 2693   | 4576   |
| 20   | 1.0   | 686       | 0.5016 | 0.5499  | 41.1    | 56.4    | 70.4     | 29.6   | 2693   | 4576   |

## Per-provenance breakdown

| provenance | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- |
| body       | 80   | 0.25  | 200       | 0.7179 | 0.7596  | 61.0    | 82.0    |
| body       | 50   | 0.25  | 200       | 0.7161 | 0.757   | 61.0    | 82.0    |
| body       | 80   | 0.5   | 200       | 0.7152 | 0.7574  | 61.0    | 81.5    |
| body       | 50   | 0.5   | 200       | 0.7127 | 0.7542  | 61.0    | 81.0    |
| body       | 80   | 0.75  | 200       | 0.7036 | 0.7484  | 59.5    | 80.5    |
| body       | 20   | 0.25  | 200       | 0.7022 | 0.7391  | 60.5    | 80.0    |
| body       | 50   | 0.75  | 200       | 0.7018 | 0.7458  | 59.5    | 80.0    |
| body       | 20   | 0.5   | 200       | 0.6967 | 0.7348  | 60.0    | 78.5    |
| body       | 20   | 0.75  | 200       | 0.687  | 0.7273  | 58.5    | 78.5    |
| body       | 80   | 1.0   | 200       | 0.6607 | 0.7157  | 53.5    | 78.0    |
| body       | 50   | 1.0   | 200       | 0.6505 | 0.7063  | 53.0    | 75.5    |
| body       | 20   | 0.0   | 200       | 0.6325 | 0.6865  | 49.5    | 75.5    |
| body       | 50   | 0.0   | 200       | 0.6233 | 0.6868  | 46.5    | 75.5    |
| body       | 80   | 0.0   | 200       | 0.619  | 0.6847  | 45.5    | 75.5    |
| body       | 20   | 1.0   | 200       | 0.6176 | 0.6736  | 50.0    | 71.5    |
| concept    | 80   | 0.0   | 200       | 0.4271 | 0.4726  | 33.0    | 51.5    |
| concept    | 50   | 0.0   | 200       | 0.4021 | 0.4417  | 31.5    | 48.0    |
| concept    | 80   | 0.25  | 200       | 0.376  | 0.4332  | 27.5    | 43.5    |
| concept    | 50   | 0.25  | 200       | 0.3589 | 0.4086  | 26.5    | 42.5    |
| concept    | 80   | 0.5   | 200       | 0.3497 | 0.4123  | 24.5    | 40.5    |
| concept    | 50   | 0.5   | 200       | 0.3371 | 0.3915  | 24.0    | 39.0    |
| concept    | 80   | 0.75  | 200       | 0.3143 | 0.3846  | 20.5    | 35.5    |
| concept    | 20   | 0.0   | 200       | 0.3057 | 0.3349  | 23.5    | 36.5    |
| concept    | 50   | 0.75  | 200       | 0.2972 | 0.3603  | 19.5    | 33.5    |
| concept    | 20   | 0.25  | 200       | 0.286  | 0.32    | 21.5    | 33.5    |
| concept    | 20   | 0.5   | 200       | 0.2739 | 0.3105  | 20.0    | 32.0    |
| concept    | 80   | 1.0   | 200       | 0.2603 | 0.3426  | 14.5    | 30.0    |
| concept    | 20   | 0.75  | 200       | 0.2537 | 0.2949  | 17.5    | 31.0    |
| concept    | 50   | 1.0   | 200       | 0.2455 | 0.3194  | 14.5    | 28.0    |
| concept    | 20   | 1.0   | 200       | 0.21   | 0.2606  | 13.0    | 26.0    |
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
| docstring  | 20   | 1.0   | 106       | 0.7262 | 0.762   | 66.0    | 78.3    |
| name       | 80   | 0.5   | 180       | 0.6053 | 0.649   | 52.8    | 65.0    |
| name       | 80   | 0.25  | 180       | 0.6    | 0.6448  | 52.2    | 64.4    |
| name       | 50   | 0.5   | 180       | 0.5987 | 0.6426  | 52.2    | 64.4    |
| name       | 50   | 0.25  | 180       | 0.595  | 0.6398  | 51.7    | 63.9    |
| name       | 20   | 0.5   | 180       | 0.5911 | 0.6291  | 52.2    | 63.3    |
| name       | 80   | 0.75  | 180       | 0.5906 | 0.6377  | 50.6    | 63.9    |
| name       | 50   | 0.75  | 180       | 0.5827 | 0.6302  | 50.0    | 62.2    |
| name       | 80   | 1.0   | 180       | 0.5816 | 0.6306  | 49.4    | 62.2    |
| name       | 50   | 1.0   | 180       | 0.5775 | 0.6263  | 48.9    | 61.7    |
| name       | 20   | 0.25  | 180       | 0.5762 | 0.6176  | 50.6    | 61.7    |
| name       | 20   | 0.75  | 180       | 0.5729 | 0.6151  | 49.4    | 61.7    |
| name       | 20   | 1.0   | 180       | 0.5646 | 0.6088  | 47.8    | 60.6    |
| name       | 80   | 0.0   | 180       | 0.5408 | 0.5996  | 43.9    | 60.0    |
| name       | 50   | 0.0   | 180       | 0.5396 | 0.5975  | 43.9    | 59.4    |
| name       | 20   | 0.0   | 180       | 0.5301 | 0.5827  | 43.3    | 59.4    |

## Per-kind breakdown

| kind       | pool | blend | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % | p50 ms |
| ---------- | ---- | ----- | --------- | ------ | ------- | ------- | ------- | ------ |
| code       | 80   | 0.5   | 119       | 0.7761 | 0.8068  | 69.7    | 84.9    | 10845  |
| code       | 50   | 0.5   | 119       | 0.7719 | 0.8015  | 69.7    | 84.0    | 6746   |
| code       | 80   | 0.25  | 119       | 0.7716 | 0.8037  | 68.1    | 85.7    | 10845  |
| code       | 80   | 0.75  | 119       | 0.7681 | 0.8007  | 68.9    | 84.0    | 10845  |
| code       | 50   | 0.25  | 119       | 0.7674 | 0.7984  | 68.1    | 84.9    | 6746   |
| code       | 50   | 0.75  | 119       | 0.7651 | 0.7963  | 68.9    | 83.2    | 6746   |
| code       | 20   | 0.5   | 119       | 0.7485 | 0.7756  | 68.1    | 80.7    | 3029   |
| code       | 20   | 0.75  | 119       | 0.7472 | 0.7744  | 68.1    | 80.7    | 3029   |
| code       | 20   | 0.25  | 119       | 0.7448 | 0.7731  | 66.4    | 82.4    | 3029   |
| code       | 80   | 1.0   | 119       | 0.7416 | 0.7803  | 65.5    | 83.2    | 10845  |
| code       | 50   | 1.0   | 119       | 0.7312 | 0.7702  | 64.7    | 80.7    | 6746   |
| code       | 20   | 1.0   | 119       | 0.7025 | 0.7398  | 62.2    | 76.5    | 3029   |
| code       | 20   | 0.0   | 119       | 0.6438 | 0.6976  | 48.7    | 79.8    | 3029   |
| code       | 50   | 0.0   | 119       | 0.6332 | 0.6976  | 45.4    | 80.7    | 6746   |
| code       | 80   | 0.0   | 119       | 0.6316 | 0.6985  | 44.5    | 81.5    | 10845  |
| concept    | 80   | 0.0   | 213       | 0.4465 | 0.4908  | 35.2    | 53.1    | 9464   |
| concept    | 50   | 0.0   | 213       | 0.4231 | 0.4617  | 33.8    | 49.8    | 6011   |
| concept    | 80   | 0.25  | 213       | 0.3951 | 0.4511  | 29.6    | 45.5    | 9464   |
| concept    | 50   | 0.25  | 213       | 0.3791 | 0.428   | 28.6    | 44.6    | 6011   |
| concept    | 80   | 0.5   | 213       | 0.3704 | 0.4315  | 26.8    | 42.7    | 9464   |
| concept    | 50   | 0.5   | 213       | 0.3585 | 0.4119  | 26.3    | 41.3    | 6011   |
| concept    | 20   | 0.0   | 213       | 0.3326 | 0.3615  | 26.3    | 39.0    | 2848   |
| concept    | 80   | 0.75  | 213       | 0.3316 | 0.4013  | 22.1    | 38.0    | 9464   |
| concept    | 50   | 0.75  | 213       | 0.3152 | 0.3782  | 21.1    | 35.7    | 6011   |
| concept    | 20   | 0.25  | 213       | 0.3106 | 0.3448  | 23.9    | 36.2    | 2848   |
| concept    | 20   | 0.5   | 213       | 0.2991 | 0.3359  | 22.5    | 34.7    | 2848   |
| concept    | 80   | 1.0   | 213       | 0.2798 | 0.3609  | 16.4    | 32.4    | 9464   |
| concept    | 20   | 0.75  | 213       | 0.2744 | 0.3168  | 19.2    | 33.3    | 2848   |
| concept    | 50   | 1.0   | 213       | 0.2655 | 0.3388  | 16.4    | 30.0    | 6011   |
| concept    | 20   | 1.0   | 213       | 0.2316 | 0.2832  | 15.0    | 28.2    | 2848   |
| identifier | 80   | 0.25  | 354       | 0.6736 | 0.7151  | 58.5    | 74.3    | 8543   |
| identifier | 80   | 0.5   | 354       | 0.6697 | 0.7123  | 57.6    | 74.3    | 8543   |
| identifier | 50   | 0.25  | 354       | 0.6686 | 0.71    | 57.9    | 74.0    | 5523   |
| identifier | 50   | 0.5   | 354       | 0.6635 | 0.7062  | 57.1    | 73.7    | 5523   |
| identifier | 80   | 0.75  | 354       | 0.6612 | 0.7055  | 56.5    | 73.4    | 8543   |
| identifier | 50   | 0.75  | 354       | 0.6538 | 0.6985  | 55.9    | 72.3    | 5523   |
| identifier | 20   | 0.5   | 354       | 0.6516 | 0.6891  | 56.8    | 71.8    | 2583   |
| identifier | 20   | 0.25  | 354       | 0.6508 | 0.6884  | 57.1    | 71.5    | 2583   |
| identifier | 20   | 0.75  | 354       | 0.6419 | 0.6815  | 55.4    | 71.2    | 2583   |
| identifier | 80   | 1.0   | 354       | 0.6317 | 0.6827  | 52.8    | 70.3    | 8543   |
| identifier | 50   | 1.0   | 354       | 0.6218 | 0.6736  | 52.0    | 68.9    | 5523   |
| identifier | 80   | 0.0   | 354       | 0.6211 | 0.675   | 51.4    | 69.5    | 8543   |
| identifier | 50   | 0.0   | 354       | 0.6194 | 0.6724  | 51.4    | 69.2    | 5523   |
| identifier | 20   | 0.0   | 354       | 0.609  | 0.6564  | 51.4    | 68.1    | 2583   |
| identifier | 20   | 1.0   | 354       | 0.5966 | 0.6465  | 49.7    | 66.7    | 2583   |

## Latency-aware selection

MRR rises with pool size, but so does latency (p50 is a function of
pool only). For each kind and overall, the cheapest (smallest-pool)
config whose MRR is within a tolerance of that scope's best, and the
MRR given up for it. Choose a pool per your latency budget; the config
is applied manually.

| kind       | tol % | pool | blend | MRR    | p50 ms | MRR vs best |
| ---------- | ----- | ---- | ----- | ------ | ------ | ----------- |
| code       | 0.5   | 80   | 0.5   | 0.7761 | 10845  | 0.0         |
| code       | 1.0   | 50   | 0.5   | 0.7719 | 6746   | 0.0042      |
| code       | 2.0   | 50   | 0.5   | 0.7719 | 6746   | 0.0042      |
| code       | 3.0   | 50   | 0.5   | 0.7719 | 6746   | 0.0042      |
| code       | 5.0   | 20   | 0.5   | 0.7485 | 3029   | 0.0275      |
| concept    | 0.5   | 80   | 0.0   | 0.4465 | 9464   | 0.0         |
| concept    | 1.0   | 80   | 0.0   | 0.4465 | 9464   | 0.0         |
| concept    | 2.0   | 80   | 0.0   | 0.4465 | 9464   | 0.0         |
| concept    | 3.0   | 80   | 0.0   | 0.4465 | 9464   | 0.0         |
| concept    | 5.0   | 80   | 0.0   | 0.4465 | 9464   | 0.0         |
| identifier | 0.5   | 80   | 0.25  | 0.6736 | 8543   | 0.0         |
| identifier | 1.0   | 50   | 0.25  | 0.6686 | 5523   | 0.005       |
| identifier | 2.0   | 50   | 0.25  | 0.6686 | 5523   | 0.005       |
| identifier | 3.0   | 50   | 0.25  | 0.6686 | 5523   | 0.005       |
| identifier | 5.0   | 20   | 0.5   | 0.6516 | 2583   | 0.022       |
| all        | 0.5   | 80   | 0.25  | 0.6041 | 9266   | 0.0         |
| all        | 1.0   | 80   | 0.25  | 0.6041 | 9266   | 0.0         |
| all        | 2.0   | 50   | 0.25  | 0.5959 | 5922   | 0.0083      |
| all        | 3.0   | 50   | 0.25  | 0.5959 | 5922   | 0.0083      |
| all        | 5.0   | 50   | 0.25  | 0.5959 | 5922   | 0.0083      |

## Per-slug breakdown (highest-MRR config: pool 80, blend 0.25)

| slug               | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ------------------ | --------- | ------ | ------- | ------- | ------- |
| anthropics__skills | 140       | 0.6963 | 0.7429  | 58.6    | 78.6    |
| astral-sh__uv      | 175       | 0.6084 | 0.6518  | 52.0    | 66.3    |
| badlogic__pi-mono  | 110       | 0.5891 | 0.6369  | 50.9    | 62.7    |
| django__django     | 161       | 0.5272 | 0.5637  | 45.3    | 60.2    |
| rbtr__rbtr         | 100       | 0.6078 | 0.66    | 49.0    | 70.0    |

## Per-language breakdown (highest-MRR config: pool 80, blend 0.25)

| language   | n_queries | MRR    | NDCG@10 | hit@1 % | hit@3 % |
| ---------- | --------- | ------ | ------- | ------- | ------- |
| bash       | 85        | 0.7206 | 0.7589  | 64.7    | 76.5    |
| css        | 80        | 0.7903 | 0.833   | 68.8    | 90.0    |
| html       | 15        | 0.354  | 0.3793  | 33.3    | 33.3    |
| javascript | 60        | 0.7744 | 0.8204  | 68.3    | 83.3    |
| json       | 75        | 0.3166 | 0.3721  | 21.3    | 37.3    |
| markdown   | 75        | 0.3888 | 0.4399  | 29.3    | 44.0    |
| plaintext  | 30        | 0.3906 | 0.4741  | 20.0    | 53.3    |
| python     | 80        | 0.7491 | 0.7772  | 68.8    | 82.5    |
| rst        | 30        | 0.4325 | 0.4632  | 40.0    | 43.3    |
| rust       | 20        | 0.6017 | 0.6502  | 50.0    | 70.0    |
| sql        | 20        | 0.6833 | 0.7371  | 55.0    | 75.0    |
| toml       | 20        | 0.65   | 0.6762  | 55.0    | 75.0    |
| typescript | 60        | 0.7414 | 0.7849  | 63.3    | 83.3    |
| yaml       | 36        | 0.4851 | 0.535   | 38.9    | 55.6    |

## Run metadata

| field             | value   |
| ----------------- | ------- |
| queries evaluated | 686     |
| configs evaluated | 15      |
| elapsed           | 13244 s |

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

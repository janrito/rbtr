# Changelog

## 2026-05-13 — Read-write workflow restructure

Restructured from a 3-step write-only pipeline to a
5-step read-write loop: read → check pending → add
comments → verify → submit.

- Added `fetch_review_comments.graphql` — reliable
  for PENDING reviews (unlike `reviewThreads`, which
  silently drops comments on newly added files)
- Workflow now requires reading context and existing
  threads before writing
- Description updated to trigger on read/resume intent

### Eval baseline (17 evals)

|                  | Checks passed | Evals fully passed |
| ---------------- | ------------- | ------------------ |
| **With skill**   | 64/66 (97%)   | 15/17              |
| **Without skill**| 59/66 (89%)   | 13/17              |

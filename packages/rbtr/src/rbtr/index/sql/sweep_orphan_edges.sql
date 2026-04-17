-- Delete edges rows whose (repo_id, commit_sha) has no matching
-- indexed_commits row.  Edges are per-commit; residue from a
-- crashed build.
DELETE FROM edges
WHERE repo_id = ?
  AND NOT EXISTS (
    SELECT 1 FROM indexed_commits ic
    WHERE ic.repo_id = edges.repo_id
      AND ic.commit_sha = edges.commit_sha
  )

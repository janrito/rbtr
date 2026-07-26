-- sqlfluff:templater:placeholder:source_id:'a'
-- sqlfluff:templater:placeholder:target_id:'b'
-- sqlfluff:templater:placeholder:kind:'imports'
SELECT
  e.source_id,
  e.target_id,
  e.kind
FROM edges AS e
INNER JOIN _snapshot_refs AS rr
  ON e.repo_id = rr.repo_id AND e.snapshot_sha = rr.snapshot_sha
WHERE
  ($source_id IS NULL OR e.source_id = $source_id)
  AND ($target_id IS NULL OR e.target_id = $target_id)
  AND ($kind IS NULL OR e.kind = $kind)

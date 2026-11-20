-- sqlfluff:templater:placeholder:steps:8
-- The places a qualified name also admits through inheritance: the scope
-- of every base of each class `_qualifiers` places, followed through
-- `inherits` edges at most $steps times.  A class is placed when its own
-- `::` chain is a place: ending with a place with no file, or equal to a
-- place in its file.
WITH RECURSIVE classes AS (
  SELECT DISTINCT
    fs.repo_id,
    fs.snapshot_sha,
    c.id AS class_id,
    fs.file_path AS class_path
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.blob_sha = fs.blob_sha
      AND c.file_language = fs.detected_language
  INNER JOIN _snapshot_refs AS rr
    ON fs.repo_id = rr.repo_id AND fs.snapshot_sha = rr.snapshot_sha
  INNER JOIN _qualifiers AS q
    ON
      (
        q.file_path IS NULL
        AND (
          c.scope || '::' || c.name = q.scope
          OR c.name = q.scope AND c.scope = ''
          OR ends_with(c.scope || '::' || c.name, '::' || q.scope)
        )
      )
      OR (
        fs.repo_id = q.repo_id
        AND fs.file_path = q.file_path
        AND (
          c.scope || '::' || c.name = q.scope
          OR c.name = q.scope AND c.scope = ''
        )
      )
  WHERE c.kind = 'class'
),

bases AS (
  SELECT
    repo_id,
    snapshot_sha,
    class_id,
    class_path,
    0 AS distance
  FROM classes
  UNION
  SELECT
    b.repo_id,
    b.snapshot_sha,
    e.target_id AS class_id,
    e.target_path AS class_path,
    b.distance + 1 AS distance
  FROM bases AS b
  INNER JOIN edges AS e
    ON
      b.repo_id = e.repo_id
      AND b.snapshot_sha = e.snapshot_sha
      AND b.class_id = e.source_id
      AND b.class_path = e.source_path
      AND e.kind = 'inherits'
  WHERE b.distance < $steps
)

SELECT DISTINCT
  b.repo_id,
  b.class_path AS file_path,
  CASE
    WHEN c.scope = '' THEN c.name
    ELSE c.scope || '::' || c.name
  END AS scope
FROM bases AS b
INNER JOIN chunks AS c ON b.class_id = c.id
WHERE b.distance > 0

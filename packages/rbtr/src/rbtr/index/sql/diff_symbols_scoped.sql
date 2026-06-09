-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:head_sha:'def'
-- sqlfluff:templater:placeholder:base_sha:'abc'
--
-- Symbol-level diff scoped to a caller-supplied set of files.
-- Identical to diff_symbols.sql but both CTEs inner-join the
-- cursor-registered `_file_paths` view, so only changes in the
-- listed files surface. See diff_symbols.sql for the diff logic.
WITH head AS (
  SELECT
    c.id,
    c.repo_id,
    c.blob_sha,
    c.file_path,
    c.kind,
    c.name,
    c.scope,
    c.language,
    c.content,
    c.line_start,
    c.line_end,
    c.metadata,
    c.embedding IS NOT NULL AS has_embedding
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.repo_id = fs.repo_id
      AND c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  INNER JOIN _file_paths AS fp
    ON c.file_path = fp.file_path
  WHERE
    fs.repo_id = $repo_id
    AND fs.commit_sha = $head_sha
    AND c.kind IN ('function', 'class', 'method')
),

base AS (
  SELECT
    c.id,
    c.repo_id,
    c.blob_sha,
    c.file_path,
    c.kind,
    c.name,
    c.scope,
    c.language,
    c.content,
    c.line_start,
    c.line_end,
    c.metadata,
    c.embedding IS NOT NULL AS has_embedding
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.repo_id = fs.repo_id
      AND c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  INNER JOIN _file_paths AS fp
    ON c.file_path = fp.file_path
  WHERE
    fs.repo_id = $repo_id
    AND fs.commit_sha = $base_sha
    AND c.kind IN ('function', 'class', 'method')
)

-- Added: present at head, absent at base.
SELECT
  'added' AS change_kind,
  h.id,
  h.repo_id,
  h.blob_sha,
  h.file_path,
  h.kind,
  h.name,
  h.scope,
  h.language,
  h.content,
  h.line_start,
  h.line_end,
  h.metadata,
  h.has_embedding
FROM head AS h
WHERE
  NOT EXISTS (
    SELECT 1 FROM base AS b
    WHERE
      b.file_path = h.file_path
      AND b.name = h.name
      AND b.scope = h.scope
  )

UNION ALL

-- Removed: present at base, absent at head.
SELECT
  'removed' AS change_kind,
  b.id,
  b.repo_id,
  b.blob_sha,
  b.file_path,
  b.kind,
  b.name,
  b.scope,
  b.language,
  b.content,
  b.line_start,
  b.line_end,
  b.metadata,
  b.has_embedding
FROM base AS b
WHERE
  NOT EXISTS (
    SELECT 1 FROM head AS h
    WHERE
      h.file_path = b.file_path
      AND h.name = b.name
      AND h.scope = b.scope
  )

UNION ALL

-- Modified: present on both sides with differing content.
SELECT
  'modified' AS change_kind,
  h.id,
  h.repo_id,
  h.blob_sha,
  h.file_path,
  h.kind,
  h.name,
  h.scope,
  h.language,
  h.content,
  h.line_start,
  h.line_end,
  h.metadata,
  h.has_embedding
FROM head AS h
INNER JOIN base AS b
  ON
    h.file_path = b.file_path
    AND h.name = b.name
    AND h.scope = b.scope
WHERE h.content <> b.content

ORDER BY change_kind, file_path, line_start

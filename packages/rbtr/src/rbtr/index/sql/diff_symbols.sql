-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:head_sha:'def'
-- sqlfluff:templater:placeholder:base_sha:'abc'
-- sqlfluff:templater:placeholder:scope_all:true
--
-- Symbol-level diff between two indexed commits in a single pass.
-- Symbol identity is (file_path, name, scope); "modified" means the
-- key exists on both sides with differing content. Each branch
-- selects one side's columns as plain references so the projection
-- matches ChunkResultRow exactly (no COALESCE aliases on keyword
-- columns like name/content). The synthetic label is `change_kind`.
--
-- The diff is optionally scoped to a set of files: when $scope_all is
-- false, both CTEs keep only rows whose file_path is present in the
-- cursor-registered `_file_paths` view (a semi-join). When true, the
-- view is ignored and every file participates.
WITH head AS (
  SELECT
    c.id,
    fs.repo_id,
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
      c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  WHERE
    fs.repo_id = $repo_id
    AND fs.commit_sha = $head_sha
    AND c.kind IN ('function', 'class', 'method')
    AND (
      $scope_all
      OR EXISTS (
        SELECT 1 FROM _file_paths AS fp
        WHERE fp.file_path = c.file_path
      )
    )
),

base AS (
  SELECT
    c.id,
    fs.repo_id,
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
      c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  WHERE
    fs.repo_id = $repo_id
    AND fs.commit_sha = $base_sha
    AND c.kind IN ('function', 'class', 'method')
    AND (
      $scope_all
      OR EXISTS (
        SELECT 1 FROM _file_paths AS fp
        WHERE fp.file_path = c.file_path
      )
    )
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

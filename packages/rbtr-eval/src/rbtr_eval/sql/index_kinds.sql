WITH chunk_counts AS (
  SELECT
    c.kind,
    COUNT(DISTINCT c.id) AS n
  FROM indexed_snapshots AS s
  INNER JOIN file_snapshots AS fs
    ON
      s.repo_id = fs.repo_id
      AND s.snapshot_sha = fs.snapshot_sha
  INNER JOIN chunks AS c
    ON
      fs.blob_sha = c.blob_sha
      AND fs.detected_language = c.file_language
  GROUP BY c.kind
),

scoped_edges AS (
  SELECT
    e.source_id,
    e.target_id
  FROM indexed_snapshots AS s
  INNER JOIN edges AS e
    ON
      s.repo_id = e.repo_id
      AND s.snapshot_sha = e.snapshot_sha
),

outbound AS (
  SELECT
    c.kind,
    COUNT(*) AS outbound_edges
  FROM scoped_edges AS e
  INNER JOIN chunks AS c ON e.source_id = c.id
  GROUP BY c.kind
),

inbound AS (
  SELECT
    c.kind,
    COUNT(*) AS inbound_edges
  FROM scoped_edges AS e
  INNER JOIN chunks AS c ON e.target_id = c.id
  GROUP BY c.kind
)

SELECT
  cc.kind,
  cc.n,
  COALESCE(o.outbound_edges, 0) AS outbound_edges,
  COALESCE(i.inbound_edges, 0) AS inbound_edges
FROM chunk_counts AS cc
LEFT JOIN outbound AS o ON cc.kind = o.kind
LEFT JOIN inbound AS i ON cc.kind = i.kind
ORDER BY cc.n DESC, cc.kind ASC

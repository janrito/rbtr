WITH chunk_counts AS (
  SELECT
    kind,
    COUNT(*) AS n
  FROM chunks
  GROUP BY kind
),

outbound AS (
  SELECT
    c.kind,
    COUNT(*) AS outbound_edges
  FROM edges AS e
  INNER JOIN chunks AS c ON e.source_id = c.id AND e.repo_id = c.repo_id
  GROUP BY c.kind
),

inbound AS (
  SELECT
    c.kind,
    COUNT(*) AS inbound_edges
  FROM edges AS e
  INNER JOIN chunks AS c ON e.target_id = c.id AND e.repo_id = c.repo_id
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
ORDER BY cc.n DESC

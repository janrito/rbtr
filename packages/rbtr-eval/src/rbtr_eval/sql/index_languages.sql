WITH chunk_counts AS (
  SELECT
    CASE WHEN language = '' THEN '(plaintext)' ELSE language END AS lang,
    COUNT(*) AS n
  FROM chunks
  GROUP BY language
),

outbound AS (
  SELECT
    CASE WHEN c.language = '' THEN '(plaintext)' ELSE c.language END AS lang,
    COUNT(*) AS outbound_edges
  FROM edges AS e
  INNER JOIN chunks AS c ON e.source_id = c.id AND e.repo_id = c.repo_id
  GROUP BY c.language
),

inbound AS (
  SELECT
    CASE WHEN c.language = '' THEN '(plaintext)' ELSE c.language END AS lang,
    COUNT(*) AS inbound_edges
  FROM edges AS e
  INNER JOIN chunks AS c ON e.target_id = c.id AND e.repo_id = c.repo_id
  GROUP BY c.language
)

SELECT
  cc.lang,
  cc.n,
  COALESCE(o.outbound_edges, 0) AS outbound_edges,
  COALESCE(i.inbound_edges, 0) AS inbound_edges
FROM chunk_counts AS cc
LEFT JOIN outbound AS o ON cc.lang = o.lang
LEFT JOIN inbound AS i ON cc.lang = i.lang
ORDER BY cc.n DESC

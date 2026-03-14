-- Check whether a history-repair incident with matching strategy
-- and fingerprint already exists for this session.
-- Params: session_id, strategy, fingerprint.
SELECT 1
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind = 'llm-history-repair'
  AND JSON_EXTRACT(data_json, '$.strategy') = ?
  AND JSON_EXTRACT(data_json, '$.fingerprint') = ?
ORDER BY rowid
LIMIT 1

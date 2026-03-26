-- Earliest created_at among active (non-compacted) message
-- fragments that are NOT in the set being compacted.
-- Params: session_id, then one ? per compact_id.
SELECT MIN(created_at) AS ts
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind IN ('request-message', 'response-message')
  AND JSON_EXTRACT(data_json, '$.compacted_by') IS NULL
  AND id NOT IN (/*compact_ids*/)

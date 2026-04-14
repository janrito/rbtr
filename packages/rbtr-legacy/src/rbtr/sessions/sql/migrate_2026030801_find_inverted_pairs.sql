-- Find response→request pairs where the response's created_at
-- is earlier than the request's (inverted persistence order).
-- Returns one row per pair with both message IDs and timestamps.
SELECT
  resp.id AS resp_id,
  resp.created_at AS resp_at,
  req.id AS req_id,
  req.created_at AS req_at
FROM fragments AS resp
INNER JOIN fragments AS req
  ON
    resp.session_id = req.session_id
    AND req.fragment_index = 0
    AND req.fragment_kind = 'request-message'
    AND req.status = 'complete'
    AND req.compacted_by IS NULL
    AND resp.created_at < req.created_at
    AND NOT EXISTS (
      SELECT 1 FROM fragments AS mid
      WHERE
        mid.session_id = resp.session_id
        AND mid.fragment_index = 0
        AND mid.fragment_kind IN ('request-message', 'response-message')
        AND mid.status = 'complete'
        AND mid.compacted_by IS NULL
        AND mid.created_at > resp.created_at
        AND mid.created_at < req.created_at
    )
WHERE
  resp.fragment_index = 0
  AND resp.fragment_kind = 'response-message'
  AND resp.status = 'complete'
  AND resp.compacted_by IS NULL;

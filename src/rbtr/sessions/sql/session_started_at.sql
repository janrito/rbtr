-- Earliest created_at for a session.
-- Params: session_id.
SELECT MIN(created_at) AS started
FROM fragments
WHERE session_id = ?;

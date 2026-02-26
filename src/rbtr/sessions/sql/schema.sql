CREATE TABLE IF NOT EXISTS fragments (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  message_id TEXT REFERENCES fragments (id) ON DELETE CASCADE,
  fragment_index INTEGER NOT NULL,
  fragment_kind TEXT NOT NULL,
  created_at TEXT NOT NULL,
  session_label TEXT,
  repo_owner TEXT,
  repo_name TEXT,
  model_name TEXT,
  review_target TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cache_read_tokens INTEGER,
  cache_write_tokens INTEGER,
  cost REAL,
  data_json TEXT,
  user_text TEXT,
  tool_name TEXT,
  compacted_by TEXT REFERENCES fragments (id) ON DELETE CASCADE,
  complete INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fragments_message_fragment
ON fragments (message_id, fragment_index);

CREATE INDEX IF NOT EXISTS idx_fragments_session_created
ON fragments (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_fragments_user_text
ON fragments (user_text)
WHERE user_text IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fragments_compacted_by
ON fragments (compacted_by)
WHERE compacted_by IS NOT NULL;

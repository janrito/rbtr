CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  session_label TEXT,
  repo_owner TEXT,
  repo_name TEXT,
  model_name TEXT,
  kind TEXT NOT NULL,
  message_json TEXT,
  user_text TEXT,
  tool_names TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cost REAL,
  compacted_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session
ON messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_user_text
ON messages (user_text) WHERE user_text IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_session_created
ON messages (session_id, created_at DESC);

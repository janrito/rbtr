INSERT INTO messages (
  id, session_id, created_at, session_label, repo_owner, repo_name,
  model_name, kind, message_json, user_text, tool_names,
  input_tokens, output_tokens, cost, compacted_by
) VALUES (
  ?, ?, ?, ?, ?, ?,
  ?, ?, ?, ?, ?,
  ?, ?, ?, ?
);

INSERT INTO fragments (
  id, session_id, message_id, fragment_index, fragment_kind, created_at,
  session_label, repo_owner, repo_name, model_name,
  input_tokens, output_tokens, cost,
  data_json, user_text, tool_name,
  compacted_by, complete
) VALUES (
  ?, ?, ?, ?, ?, ?,
  ?, ?, ?, ?,
  ?, ?, ?,
  ?, ?, ?,
  ?, ?
);

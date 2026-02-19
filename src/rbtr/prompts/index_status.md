# Index status

{% if status == "ready" %}
The code index is ready. Use {{ tool_list }} to ground your
analysis in the actual codebase.
{% elif status == "building" %}
⚠ IMPORTANT: The code index is still building in the background.
You MUST tell the user this before anything else.
Index-based tools ({{ tool_list }}) are NOT available yet and
will fail if called. Acknowledge the limitation, answer with
what you can, and suggest the user ask again once indexing
completes.
{% endif %}

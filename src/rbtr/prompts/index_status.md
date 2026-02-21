# Index status

{% if status == "ready" %}
The code index is ready. Use {{ tool_list }} to ground your
analysis in the actual codebase.
{% elif status == "building" %}
⚠ IMPORTANT: The code index is still building in the background.
Index-based tools ({{ tool_list }}) are NOT available yet and
will fail if called. Tell the user the index is still building,
then STOP and wait for their next message. Do NOT attempt to
answer, use tools, or continue until the user responds.
{% endif %}

# Compact

Summarise the following conversation history concisely.
Preserve:

- The user's overall goal and current task
- Key decisions made and their rationale
- Files read, modified, or discussed (with paths)
- Important findings, errors, or blockers
- The current state of progress

Do NOT include pleasantries, redundant context, or raw file
contents. Focus on what a future assistant turn would need to
continue the work effectively.
{% if extra_instructions %}

Additional instructions: {{ extra_instructions }}
{% endif %}

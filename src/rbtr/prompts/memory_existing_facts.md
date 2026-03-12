## Existing facts

These facts are already stored. Do NOT extract duplicates.
If a fact is **confirmed** by the conversation, mark it
`confirm`. If **contradicted or outdated**, mark it
`supersede` and provide the replacement.

{% for fact in facts %}- {{ fact }}
{% endfor %}

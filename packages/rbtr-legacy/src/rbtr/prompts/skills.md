The following skills provide specialized instructions for
specific tasks. Use `read_file` to load a skill's file when
the task matches its description. When a skill file references
a relative path, resolve it against the skill directory (parent
of SKILL.md / dirname of the path) and use that absolute path
in tool commands.

<available_skills>
{% for skill in skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{ skill.file_path }}</location>
  </skill>
{% endfor %}
</available_skills>

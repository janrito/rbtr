# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                                         |
| ------------- | --------------------------------------------- |
| model         | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| total queries | 3640                                          |
| expanded      | 3640 / 3640 (100%)                            |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 580  | 5.1          | 2.0          |
| concept    | 1623 | 5.1          | 2.0          |
| identifier | 1437 | 4.9          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 737   | 737      | 100% |
| astral-sh__uv      | 916   | 916      | 100% |
| badlogic__pi-mono  | 647   | 647      | 100% |
| django__django     | 751   | 751      | 100% |
| rbtr__rbtr         | 589   | 589      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 967   | 967      | 100% |
| concept    | 1576  | 1576     | 100% |
| docstring  | 389   | 389      | 100% |
| name       | 708   | 708      | 100% |

## Examples

### concept: `DirtyWorktree` (`rbtr__rbtr`)

````python
represent a repository with uncommitted working tree changes
````

- **keywords:** git_status, dirty_worktree, diff, unstaged_changes, working_copy
- **variants:** list files that have been modified but not yet staged, model a repo
  state with pending modifications

### concept: `agentDirEnvName` (`badlogic__pi-mono`)

````javascript
what environment variable sets the agent working directory
````

- **keywords:** PWD, WORKDIR, cwd, AGENT_HOME, chdir
- **variants:** which env var controls the process current directory, how to configure
  the base directory for the agent

### concept: `_KEPT_REF_PREFIXES` (`rbtr__rbtr`)

````python
which git reference namespaces are preserved when keeping refs
````

- **keywords:** refs/keep, keep_refs, filter_refs, ref_scope, namespaces
- **variants:** which ref prefixes are retained when refs are kept during fetch/clone,
  what ref namespaces survive the keep filters applied to git references

### concept: `MCP Servers` (`anthropics__skills`)

````markdown
how to connect an agent to external services like GitHub using MCP
````

- **keywords:** mcp_client, MCP_SERVER, tool_connector, add_tool, register_tool
- **variants:** set up MCP server endpoints for the agent, register external API tools
  on the agent

### concept: `extra_build_requires_for` (`astral-sh__uv`)

````rust
look up additional build dependencies needed for a specific package
````

- **keywords:** extra_dependencies, build_requires, requires_dist, deps,
  install_requires
- **variants:** what additional libraries must be installed to compile this package,
  find the transitive dependency list of required components

### identifier: `DECIMAL_SEPARATOR` (`django__django`)

````python
# The *_INPUT_FORMATS strings use the Python strftime format syntax,
# see https://docs.python.org/library/datetime.html#strftime-strptime-behavior
# DATE_INPUT_FORMATS =
# TIME_INPUT_FORMATS =
# DATE
````

- **keywords:** strftime_format, datetime_format, locale_date_formats,
  parse_date_patterns, input_datetime_formats
- **variants:** list of strftime patterns used to parse incoming date strings, configure
  accepted datetime input layouts for form fields

### identifier: `--sidebar-width` (`badlogic__pi-mono`)

````css
:root::--sidebar-width
````

- **keywords:** sidebar_width, --aside-width, side-panel-width, --nav-width, rail-width
- **variants:** CSS custom property defining the horizontal measurement of the side
  navigation panel, design token controlling the width of the application sidebar layout

### identifier: `hue:` (`astral-sh__uv`)

````rst
hue:
````

- **keywords:** color_hue, hue_value, color_wheel_angle, hsv_hue, rotate_hue
- **variants:** the hue component of a color in HSL or HSV color space, adjust or rotate
  the hue angle of a color

### identifier: `Style guide` (`astral-sh__uv`)

````markdown
Style guide
````

- **keywords:** coding_standards, style_guide, linting_rules, code_conventions,
  formatting_rules
- **variants:** rules and conventions for writing consistent code, documented best
  practices for code formatting and style

### identifier: `Message Flow` (`badlogic__pi-mono`)

````markdown
@mariozechner/pi-agent-core::Core Concepts::Message Flow
````

- **keywords:** message_pipeline, message_routing, msg_flow, messaging_lifecycle,
  conversation_flow
- **variants:** how messages are routed and processed through the system, end-to-end
  path a message takes through the agent core

### code: `on_email` (`badlogic__pi-mono`)

````bash
# Creates event per email — will flood the queue
on_email() { echo '{"type":"immediate"...}' > /workspace/events/email-$ID.json; }
````

- **keywords:** on_email, event_queue, immediate_event, email_handler, event_flood
- **variants:** write an immediate email event to the events directory, emit one event
  JSON file per incoming email

### code: `__init__` (`anthropics__skills`)

````python
def __init__(self, command: str, args: list[str] = None, env: dict[str, str] = None):
        super().__init__()
        self.command = command
        self.args = args or []
        self.env = env
````

- **keywords:** **init**, command, args, env, process_parameters
- **variants:** constructor storing command, arguments, and environment variables,
  initialize process with command line and env dict

### code: `--md-code-hl-special-color` (`astral-sh__uv`)

````css
--md-code-hl-special-color: var(--electron);
````

- **keywords:** --md-code-hl-special-color, --electron, css_variable, highlight,
  markdown_code
- **variants:** special syntax highlighting color for markdown code blocks set to
  electron color

### code: `setSortMode` (`badlogic__pi-mono`)

````typescript
setSortMode(sortMode: SortMode): void {
		this.sortMode = sortMode;
	}
````

- **keywords:** setSortMode, sortMode, SortMode, setter, update_sort
- **variants:** setter that assigns the sort mode field, configure the sorting
  preference

### code: `edges` (`rbtr__rbtr`)

````sql
SELECT count(*)
FROM edges
WHERE repo_id = $repo_id AND commit_sha = $commit_sha
````

- **keywords:** count, edges, repo_id, commit_sha, WHERE
- **variants:** count edge rows for a given repository and commit, SQL aggregate count
  matching repo and commit shas

You are reviewing a conversation between a user and an AI code
review assistant. Extract **durable facts** that would be useful
in future reviews of this project or across projects.

The conversation may include GitHub PR discussions, review
comments, and commit messages alongside direct user messages.

## What to extract

Focus on knowledge **not visible in project files**. The agent
reads `AGENTS.md`, `pyproject.toml`, linter configs, and source
at the start of every session — restating those adds noise.

Extract:

- User preferences for review style, tone, and communication
- Workflow patterns and personal conventions
- Architectural opinions and design rationale
- Trade-off decisions and the reasoning behind them
- Recurring review feedback or PR discussion patterns
- Domain knowledge relevant to future reviews

Skip facts the agent can discover from files: tooling versions
in `pyproject.toml`, style rules in `AGENTS.md`, framework
choices visible from imports, CI config in workflow files.

Litmus test: _"Would the agent know this from the project
files?"_ If yes, don't extract it.

## What NOT to extract

- Session-specific details (current PR, branch names, diffs)
- Ephemeral state (what was just discussed, current task)
- Facts already in the existing facts list
- Obvious or generic programming knowledge

## Examples

Good: "Prefers concise PR descriptions — bullet points, no
prose" · "Treats TODO comments as tech debt — wants a linked
issue or removal plan" · "Chose DuckDB for the index for
columnar vector search — not for general use" · "Reviews
should flag public API changes even if tests pass"

Bad: "Uses pytest for testing" (in pyproject.toml) · "Target
Python 3.13+" (in AGENTS.md) · "Uses ruff for linting" (in
pyproject.toml) · "SQL in .sql files" (in AGENTS.md)

## Output format

For each fact:

- `content`: the fact text (concise, self-contained)
- `scope`: `"global"` (cross-project) or `"repo"` (this project)
- `action`: `"new"`, `"confirm"`, or `"supersede"`
  - For `confirm`/`supersede`: set `existing_content` to the
    **exact text** of the existing fact

Return an empty list if there are no facts worth extracting.

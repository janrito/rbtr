---
name: review-github-pr
description: >-
  Manage GitHub pull request reviews using the GraphQL
  API via `gh`. Use whenever the user asks to review a
  PR, read or check existing review comments, resume a
  pending review, add review comments, post inline
  suggestions, approve or request changes on a pull
  request, reply to review threads, submit review
  feedback, or interact with GitHub's code review
  system in any way.
compatibility: Requires `gh` (GitHub CLI, authenticated) and `jq`.
---

# Managing a GitHub PR review

This skill covers the mechanics of working with
reviews on GitHub via `gh api graphql`. It is not a
guide on how to write good reviews — it's a reference
for reading, posting, and verifying review comments
correctly.

All GraphQL queries live in `queries/` alongside this
file. Read the `.graphql` file, fill in variables, and
pass it to `gh api graphql`. Never embed GraphQL inline
or from memory — the query files are validated against
GitHub's schema and are the single source of truth.

## Workflow

### 1. Read PR context

Fetch the diff and understand the PR before commenting:

```bash
# Unified diff — read this to pick valid line numbers
gh pr diff <number> -R owner/repo
```

Fetch the full PR — title, description, and all
existing review threads:

```text
→ read queries/fetch_pr.graphql
```

| Variable | Type     | Flag | Value            |
| -------- | -------- | ---- | ---------------- |
| `owner`  | `String` | `-f` | Repository owner |
| `name`   | `String` | `-f` | Repository name  |
| `number` | `Int`    | `-F` | PR number        |

Fetch node IDs and pending review status:

```text
→ read queries/pr_id.graphql
```

| Variable | Type     | Flag | Value            |
| -------- | -------- | ---- | ---------------- |
| `owner`  | `String` | `-f` | Repository owner |
| `name`   | `String` | `-f` | Repository name  |
| `number` | `Int`    | `-F` | PR number        |

From `fetch_pr.graphql` you get:

- `pullRequest.title` / `pullRequest.body` — the PR
  description
- `pullRequest.headRefOid` — the head commit SHA
- `reviewThreads` — all existing review threads

From `pr_id.graphql` you get:

- `pullRequest.id` — the PR node ID (for mutations)
- `pullRequest.reviews.nodes[].id` — pending review
  node IDs (if any exist)

Read the description and existing threads. Understand
what's already been discussed before writing anything.

### 2. Check pending review state

If `pr_id.graphql` shows a pending review, fetch its
comments directly:

```text
→ read queries/fetch_review_comments.graphql
```

| Variable   | Type | Flag | Value                               |
| ---------- | ---- | ---- | ----------------------------------- |
| `reviewId` | `ID` | `-f` | Review node ID from `pr_id.graphql` |

Do not rely on `reviewThreads` from `fetch_pr.graphql`
for pending review comments — it can silently drop
them, especially on newly added files.

If no pending review exists, skip to step 3.

### 3. Add comments

**If no pending review exists** — create one with all
initial comments as threads:

```text
→ read queries/add_review.graphql
```

| Variable        | Type                             | Flag | Value                        |
| --------------- | -------------------------------- | ---- | ---------------------------- |
| `pullRequestId` | `ID`                             | `-f` | PR node ID from step 1       |
| `commitOID`     | `GitObjectID`                    | `-f` | Head SHA from step 1         |
| `body`          | `String`                         | `-f` | Review summary (markdown)    |
| `threads`       | `[DraftPullRequestReviewThread]` | `-f` | JSON array of thread objects |

Each thread object: `{"path": "...", "line": N, "body": "..."}`.
Optional fields: `side` (default `RIGHT`), `startLine`,
`startSide` (for multi-line).

**If a pending review already exists** — add comments
one at a time:

```text
→ read queries/add_comment.graphql
```

| Variable              | Type       | Flag | Value                        |
| --------------------- | ---------- | ---- | ---------------------------- |
| `pullRequestReviewId` | `ID`       | `-f` | Pending review node ID       |
| `body`                | `String`   | `-f` | Comment body (markdown)      |
| `path`                | `String`   | `-f` | File path relative to root   |
| `line`                | `Int`      | `-F` | Line number (see below)      |
| `side`                | `DiffSide` | `-f` | `RIGHT` (default) or `LEFT`  |

If an existing thread already covers the same concern,
reply to it with `add_thread_reply.graphql` (see
[references/graphql-api.md](references/graphql-api.md))
rather than starting a new thread.

If the review spans an extended period, re-fetch
threads with `fetch_pr.graphql` between additions to
catch new activity from other reviewers.

### 4. Verify before submitting

Before submitting, always:

1. Re-fetch existing threads with `fetch_pr.graphql`
   to catch new activity since you started.
2. Re-fetch your pending review with
   `fetch_review_comments.graphql` to verify all your
   comments are present.

If the counts don't match what you expect, investigate
before submitting.

### 5. Submit the review

```text
→ read queries/submit_review.graphql
```

| Variable              | Type                     | Flag | Value                                      |
| --------------------- | ------------------------ | ---- | ------------------------------------------ |
| `pullRequestReviewId` | `ID`                     | `-f` | Pending review node ID                     |
| `event`               | `PullRequestReviewEvent` | `-f` | `COMMENT`, `APPROVE`, or `REQUEST_CHANGES` |

For additional operations (update, delete, reply,
resolve threads, fetch existing threads), see
[references/graphql-api.md](references/graphql-api.md).

## Line targeting

`line` is a **1-indexed file line number**, not a diff
position. Only lines inside diff hunks are commentable.

Read the diff with `gh pr diff <number>` and parse the
hunk headers to identify valid lines.

### Which lines are commentable

Given a hunk `@@ -10,5 +12,7 @@`:

| Diff line  | LEFT (old) | RIGHT (new) | Commentable on |
| ---------- | ---------- | ----------- | -------------- |
| `context`  | 10         | 12          | LEFT or RIGHT  |
| `-removed` | 11         | —           | LEFT only      |
| `+added`   | —          | 13          | RIGHT only     |
| `context`  | 12         | 14          | LEFT or RIGHT  |
| `+added`   | —          | 15          | RIGHT only     |
| `+added`   | —          | 16          | RIGHT only     |
| `context`  | 13         | 17          | LEFT or RIGHT  |

- **`+` lines** → comment on RIGHT with the new-file
  line number.
- **`-` lines** → comment on LEFT with the old-file
  line number.
- **Context lines** (space prefix) → either side.
- **Lines outside hunks** → not commentable.

### Multi-line comments

Set `startLine` and `line` to define an inclusive range.
Both must be within the same diff hunk on the same side.
Set `startSide` to match `side`.

### File-level comments

To comment on a file without targeting a line, omit
`line` and set `subjectType: FILE` in the thread object.

### Renamed files

When a file is renamed, the diff shows
`rename from old.py` / `rename to new.py`. Use the
**new** path (the `b/` side) as `path`.

### Binary files

Binary files show `Binary files differ` with no hunks.
They have no commentable lines — use a file-level
comment instead.

## Suggestion blocks

Suggestion blocks are GitHub-flavoured markdown, not a
GraphQL feature. They go inside the comment `body`.

### Single-line suggestion

Target one line. The suggestion replaces that line:

````markdown
```suggestion
replacement code here
```
````

### Multi-line suggestion

The comment must span the range being replaced. Set
`startLine` to the first line and `line` to the last
line of the range. The suggestion body replaces the
entire range:

````markdown
```suggestion
replacement line 1
replacement line 2
```
````

Both `startLine` and `line` must be within the diff
on the same side.

## Key patterns

### Node IDs

All mutation inputs use GraphQL **node IDs** — base64
strings like `PR_kwDO...`. Get them from the `id`
field in query responses. Never use numeric
`databaseId` — it is deprecated and will not work
in mutations.

### `gh api graphql` variable flags

- `-f key=value` for **string** variables (IDs, paths,
  bodies, enum values like `RIGHT` or `COMMENT`)
- `-F key=value` for **typed** variables (integers,
  booleans — e.g. `-F number=42`)
- `--jq '.data.field'` to extract specific fields

### Cross-repo reviews

When reviewing a PR on a different repo, pass
`-R owner/repo` to `gh pr diff` and set the `owner`
and `name` GraphQL variables accordingly.

### Pending review visibility

Comments on a PENDING review are only visible to the
author. Other users cannot see them until the review
is submitted.

`reviewThreads` can silently omit pending comments,
especially on newly added files. Use
`fetch_review_comments.graphql` to verify pending
comments — see workflow step 2.

# Additional GraphQL operations

These cover operations beyond the core review workflow
in SKILL.md. Each section names a query file in
`queries/` — read it for the exact GraphQL syntax.

## Reply to an existing thread

Add a reply to a conversation thread rather than
starting a new one.

```text
→ read queries/add_thread_reply.graphql
```

| Variable                      | Type     | Flag | Value                     |
| ----------------------------- | -------- | ---- | ------------------------- |
| `pullRequestReviewThreadId`   | `ID`     | `-f` | Thread node ID            |
| `body`                        | `String` | `-f` | Reply body (markdown)     |

Get the thread node ID from the `id` field in a
`reviewThreads` query (see "Fetch existing threads"
below).

## Update a comment

Edit the body of an existing review comment.

```text
→ read queries/update_comment.graphql
```

| Variable                       | Type     | Flag | Value               |
| ------------------------------ | -------- | ---- | ------------------- |
| `pullRequestReviewCommentId`   | `ID`     | `-f` | Comment node ID     |
| `body`                         | `String` | `-f` | New body (markdown) |

## Delete a comment

Remove a single comment from a review.

```text
→ read queries/delete_comment.graphql
```

| Variable | Type | Flag | Value           |
| -------- | ---- | ---- | --------------- |
| `id`     | `ID` | `-f` | Comment node ID |

## Delete an entire pending review

Discard all pending comments at once. This cannot be
undone.

```text
→ read queries/delete_review.graphql
```

| Variable              | Type | Flag | Value              |
| --------------------- | ---- | ---- | ------------------ |
| `pullRequestReviewId` | `ID` | `-f` | Review node ID     |

## Fetch existing review threads

Read the full conversation on a PR — all threads from
all reviewers, including resolved and outdated ones.
Useful for understanding context before posting.

```text
→ read queries/fetch_pr.graphql
```

| Variable | Type     | Flag | Value            |
| -------- | -------- | ---- | ---------------- |
| `owner`  | `String` | `-f` | Repository owner |
| `name`   | `String` | `-f` | Repository name  |
| `number` | `Int`    | `-F` | PR number        |

The response includes `reviewThreads.nodes[]` with
`id`, `path`, `line`, `diffSide`, `isResolved`,
`isOutdated`, and nested `comments.nodes[]` with
`id`, `body`, `author.login`, `createdAt`.

`reviewThreads` aggregates across all reviews but
may miss pending comments on newly added files. To
read a specific review's comments reliably, use
`fetch_review_comments.graphql` (see next section).

## Fetch comments from a specific review

Read all comments belonging to a single review —
reliable for both PENDING and submitted reviews.
Use this instead of `reviewThreads` when you need
the complete set of comments from a known review.

```text
→ read queries/fetch_review_comments.graphql
```

| Variable   | Type | Flag | Value                               |
| ---------- | ---- | ---- | ----------------------------------- |
| `reviewId` | `ID` | `-f` | Review node ID from `pr_id.graphql` |

The response includes `comments.nodes[]` with `id`,
`fullDatabaseId`, `body`, `path`, `line`,
`createdAt`, plus `totalCount` and pagination info.

Typical workflow: run `pr_id.graphql` to find pending
review IDs, then `fetch_review_comments.graphql` on
each to get the full comment list.

## Resolve a thread

Mark a review thread as resolved.

```text
→ read queries/resolve_thread.graphql
```

| Variable   | Type | Flag | Value          |
| ---------- | ---- | ---- | -------------- |
| `threadId` | `ID` | `-f` | Thread node ID |

## Error reference

### `pull_request_review_thread.line must be part of the diff`

The `line` number is not within a diff hunk for the
given `path` and `side`. Re-read the diff with
`gh pr diff <number>` and pick a line that appears
inside a hunk. See the line targeting section in
SKILL.md for which lines are commentable.

### `Could not resolve to a node with the global id of 'NNNN'`

A numeric database ID was passed where a GraphQL node
ID is expected. Node IDs are base64 strings like
`PR_kwDO...`. Use the `id` field from query responses,
never `databaseId` or numeric IDs.

### `gh: authentication required`

The `gh` CLI is not authenticated. Run `gh auth login`
or check that `GH_TOKEN` / `GITHUB_TOKEN` is set.

### `Resource not accessible by integration`

The authenticated token does not have permission to
post reviews on this repository. Check the token's
scopes — it needs `repo` access for private repos.

### GraphQL errors in response body

If `gh api graphql` returns `{"errors": [...]}` with a
zero exit code, the request reached GitHub but the
query was invalid. Read the `message` field in each
error object for the specific issue.

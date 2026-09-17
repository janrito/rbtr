# review-github-pr

A [pi] skill that drives GitHub pull request reviews through
`gh api graphql`. It gives the agent the mechanics of reviewing:
reading a PR and its existing threads, opening a pending review,
posting inline comments and suggestions, replying to and resolving
threads, and submitting or discarding the review.

[pi]: https://github.com/badlogic/pi-mono

## Install

```bash
pi install npm:@janrito/skill-rbtr-review-github-pr
```

Requires the [GitHub CLI][gh] (authenticated) and [jq].

[gh]: https://cli.github.com
[jq]: https://jqlang.github.io/jq/

## What it contains

`SKILL.md` is the instruction set the agent reads. Beside it,
`queries/` holds every GraphQL document the skill uses — one file
per operation, from `fetch_pr.graphql` through `submit_review.graphql`.
The agent reads a query file and passes it to `gh api graphql` rather
than writing GraphQL from memory, so the wire format is always a file
that exists in the package.

Those files are checked against GitHub's published schema in CI
(`just validate-graphql`), which is what stops a drifted field name
from reaching a review. `references/graphql-api.md` covers the parts
of the API the skill leans on.

## Licence

MIT. Part of [rbtr]; issues and source live there.

[rbtr]: https://github.com/janrito/rbtr

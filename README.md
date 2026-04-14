# rbtr

Monorepo for rbtr — a language-agnostic code review tool.

## Packages

| Package                                | Language | Description                                 |
| -------------------------------------- | -------- | ------------------------------------------- |
| [`rbtr-legacy`](packages/rbtr-legacy/) | Python   | Original monolithic rbtr (being decomposed) |

## Development

```bash
# Install dependencies
uv sync --extra languages

# Run all checks (lint, typecheck, test)
just check

# Format
just fmt
```

## Migration

This repo is being restructured from a single Python package into
a multi-package workspace. See
[`todo/TODO-grand-migration-plan.md`](todo/TODO-grand-migration-plan.md)
for the full plan.

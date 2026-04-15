# rbtr

Monorepo for rbtr — a language-agnostic code review tool.

## Packages

| Package                                | Language   | Description                                 |
| -------------------------------------- | ---------- | ------------------------------------------- |
| [`rbtr`](packages/rbtr/)               | Python     | Structural code index (CLI + library)       |
| [`pi-rbtr`](packages/pi-rbtr/)         | TypeScript | Pi extension for the code index             |
| [`rbtr-legacy`](packages/rbtr-legacy/) | Python     | Original monolithic rbtr (being decomposed) |

## Development

```bash
# Python dependencies
uv sync --extra languages

# TypeScript dependencies
bun install

# Run all checks (lint, typecheck, test — both languages)
just check

# Format
just fmt
```

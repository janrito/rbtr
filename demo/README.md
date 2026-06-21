# rbtr demos

Scripted terminal recordings built with [VHS].
Each `.tape` file defines the commands, timing, and
theme — `vhs` runs them for real and captures the
output as a GIF.

[VHS]: https://github.com/charmbracelet/vhs

## Prerequisites

```bash
brew install vhs            # recorder (uses ffmpeg + ttyd)
uv tool install rbtr        # rbtr itself
```

## Recording

```bash
just demo                              # record all
just demo-one demo/index-search.tape   # record one
```

Output lands in `demo/output/` (checked into git).

## Demos

| Tape                      | Duration | What it shows                              |
| ------------------------- | -------- | ------------------------------------------ |
| `index-search.tape`       | ~30s     | Three search modes + symbol retrieval      |
| `structural-nav.tape`     | ~35s     | File outline, find-refs, structural diff   |
| `agent-integration.tape`  | ~75s     | pi session — LLM navigating with rbtr      |

## Before recording

- **Demo 1** — ensure the daemon is running: `rbtr index`.
- **Demo 2** — watch main so `changed-symbols` works:
  `rbtr index main`.
- **Demo 3** — the pi extension is loaded via `-e` flag
  in the tape itself.

## Embedding in the README

The GIFs in `demo/output/` are embedded in the top-level
`README.md` via relative paths.

See the [VHS command reference][vhs-docs] for the full
tape syntax.

[vhs-docs]: https://github.com/charmbracelet/vhs#vhs-command-reference

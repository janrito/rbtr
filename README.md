# rbtr

Interactive LLM-powered PR review workbench for the terminal.

rbtr runs inside a git repository, connects to GitHub, and gives you an interactive TUI for reviewing pull requests and branches. An LLM helps you understand changes, trace impact, and build a narrative — but you make all the decisions about what to comment on the PR.

## Install

Requires Python ≥ 3.13.

```bash
uv pip install -e .
```

## Usage

```bash
# Launch in the current repo
rbtr

# Jump straight to a PR
rbtr 42
```

The TUI opens immediately. On startup it validates the repo, authenticates with GitHub (if a token is stored), and lists open PRs and branches.

### Input modes

| Prefix   | What it does              | Example                 |
| -------- | ------------------------- | ----------------------- |
| _(none)_ | Send a message to the LLM | `explain this change`   |
| `/`      | Run an internal command   | `/review 42`            |
| `!`      | Run a shell command       | `!git log --oneline -5` |

### Commands

| Command        | Description                             |
| -------------- | --------------------------------------- |
| `/help`        | Show available commands                 |
| `/review`      | List open PRs and branches              |
| `/review <id>` | Select a PR by number or branch by name |
| `/login`       | Authenticate with GitHub (device flow)  |
| `/quit`        | Exit (also `/q`)                        |

### Authentication

rbtr uses GitHub's OAuth device flow. Run `/login` and follow the instructions — open the URL in your browser and enter the code. The token is stored at `~/.config/rbtr/token`.

If a token is already stored from a previous session, rbtr uses it automatically on startup.

## Architecture

### Engine / UI separation

Execution (`Engine`) and display (`UI`) are fully separated. They communicate through a `queue.Queue[Event]` of typed Pydantic models defined in `events.py`. The Engine emits events (`TaskStarted`, `Output`, `TableOutput`, `TaskFinished`, etc.) — it never touches the display. The UI consumes events and renders them — it never runs commands or does I/O. A slow command doesn't block the UI. UI rendering doesn't delay execution.

### Rich Live TUI

All rendering goes through Rich (`Console`, `Live`, `Text`, `Table`, `Markdown`, `Rule`). One `Live` context runs for the entire session. The main thread owns the Live loop — it polls for input, drains engine events, and triggers completion. Refresh rate and poll interval are defined by `REFRESH_PER_SECOND` and `POLL_INTERVAL` in `constants.py`, targeting worst-case keystroke-to-display latency under 100ms.

### Input handling (prompt_toolkit headless)

Input is handled by prompt_toolkit running headlessly — it never writes to the terminal. `Vt100Parser` parses raw keystrokes from a daemon reader thread. `Buffer(multiline=True)` manages the editing state (cursor, undo, word boundaries). The main thread reads `InputState.text` and `.cursor` for Rich rendering.

Key bindings: Enter submits, Alt+Enter inserts a newline, Up/Down navigate within multiline text (history cycling at document boundaries), standard readline shortcuts (Ctrl+A/E/W/U/K). History persists to `~/.config/rbtr/history` on every submit.

### Threading model

Tasks (commands, shell, LLM, setup) run in daemon threads via `_start_task()`. The task thread emits events to the queue. The main loop drains events, updates the active panel, and when the task finishes, prints the finalized panel to native scrollback.

### Native terminal scrollback

Finalized content (completed panels, input echoes) is printed to the terminal's native scrollback via `console.print()`. The Live region stays small — only the active task panel + input chrome. The user scrolls back through the entire session using their terminal's normal scrolling (Shift+PgUp, mouse wheel, tmux, etc.).

### History blocks

Every history block is built by `_history_panel(variant, content)`. Variants (`input`, `active`, `succeeded`, `failed`) are defined in `_HISTORY_STYLES`. Blocks use background-colour bands via `Padding` instead of `Panel` borders — this keeps copy-paste clean (no box-drawing characters).

### Visual conventions

- **User input**: bold cyan `> text`
- **System output**: dim style
- **LLM output**: rendered as Markdown
- **Errors**: bold red. **Warnings**: yellow.
- **Links**: Rich markup with OSC 8 terminal hyperlinks

### Shell completion

Tab completion for `!commands` uses a three-tier pipeline (defined in `input.py`, run in a background thread):

1. **Bash programmable completion** — sources bash-completion scripts via `bash -c` (non-interactive, no terminal parsing). Handles subcommands (`git status`), flags (`--verbose`), dynamic values (branch names, remotes). Searches well-known directories across Homebrew, apt, Nix, MacPorts, Snap.
2. **Filesystem path** — deterministic `os.listdir` for file/directory arguments.
3. **PATH executable search** — for command names (first word).

Each tier falls through on empty results. Single match auto-accepts; multiple matches extend the common prefix and show a cycling menu.

### Shell output truncation

Shell output is truncated after `SHELL_MAX_LINES`. When truncated, the panel shows `… N more lines (ctrl+o to expand)`. Ctrl+O prints the full output. Only the most recent shell output is retained for expansion.

### GitHub integration

Authentication (device flow) and API operations live in `src/rbtr/github/`. Auth functions are public primitives. Client functions receive a `Github` instance as a parameter. httpx is used for direct HTTP (e.g., device flow); PyGithub handles its own HTTP internally.

## Development

Requires [just](https://github.com/casey/just) as a command runner.

```bash
# Install dev dependencies
uv sync

# Run all checks (lint + typecheck + test)
just check

# Individual steps
just lint          # ruff check + format check
just typecheck     # mypy
just test          # pytest

# Auto-fix lint issues and reformat
just fmt

# Build the package
just build
```

### Release workflow

```bash
just pre-release   # bump to next pre-release (e.g. 2026.2.1-dev0 → dev1, or 2026.2.0 → 2026.2.1-dev0)
just release       # bump to stable (e.g. 2026.2.1-dev0 → 2026.2.1)
```

Both release commands also run `just build` automatically. Pass extra flags with `just release --dry-run`.

## Project structure

```sh
src/rbtr/
├── __init__.py          # RbtrError
├── cli.py               # Entry point: parse args, launch TUI
├── tui.py               # Rich Live app: Engine (daemon threads) + UI (main thread)
├── input.py             # Input handling (prompt_toolkit headless) + completion pipeline
├── events.py            # Typed Pydantic events for Engine→UI communication
├── constants.py         # All constants
├── styles.py            # Rich theme + style constants
├── models.py            # PRSummary, BranchSummary
├── repo.py              # pygit2 operations
└── github/
    ├── auth.py          # OAuth device flow
    └── client.py        # GitHub API operations
```

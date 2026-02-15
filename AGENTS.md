# AGENTS.md

Agent rules. Short, imperative. See README.md for architecture rationale.

## Naming

- The project is **rbtr** — always lowercase.

## UI

- Rich owns all rendering. No Textual, curses, or raw ANSI escape codes (`\033[...`).
- prompt_toolkit is used **headlessly** for input handling — `Vt100Parser` for keystroke parsing, `Buffer` for editing state. It must never write to the terminal or render anything.
- No raw ANSI escape codes (`\033[...`). All styling through Rich.
- **Panel lifecycle**: Active panel (in-progress) → scrollback or pending. When a task finishes, if all output fits (not truncated), the panel goes **directly to scrollback** — immediately selectable/copyable. Only panels with truncated output stay **pending** in Live so Ctrl+O can expand them. `_finalize_pending` prints the pending panel to scrollback when the user submits new input or a new task finishes. Ctrl+O rewrites the pending panel in-place in Live; once expanded it is immediately finalized to scrollback (large content must not stay in Live).
- **Command queuing**: Commands submitted while a task is active go into `_pending_commands`. They are NOT echoed to scrollback until dispatched — echoing early would place the input panel above the still-running task's output. A `queued` panel (dark background, muted text) appears in Live between the active panel and the input chrome showing what's waiting. Cancel clears the queue.
- **Live ↔ scrollback print ordering**: `Live.update()` only stores the new renderable — it does NOT redraw. Before printing to scrollback via `console.print()`, you MUST call `_live.update(self._render_view(), refresh=True)` so Live physically shrinks on screen first. Without `refresh=True` the terminal still reserves space for the old (large) Live renderable, and the input ends up in the middle of the viewport with empty space below. This applies everywhere content moves from Live to scrollback: `_finalize_pending`, `TaskFinished`, or any future path that clears Live and prints.
- History blocks use background-colour bands via `Padding`, not `Panel` borders (copy-paste friendly).
- History panel content must be copy-paste identical to real command output. No artificial blank lines inside the bg band. Visual spacing between panels uses margin (empty lines outside the bg band via `_print_to_scrollback`), never extra padding.
- All history blocks go through `_history_panel(variant, content)`. No ad-hoc `Padding(...)`.
- Never drop information from carefully designed messages (counts, labels, structure). When moving a message between Engine/UI layers, preserve its full content.
- Panel variants: `input`, `active`, `succeeded`, `failed`, `queued`. To add a variant: add to `_HISTORY_STYLES`, extend the `Literal` type, add styles to `styles.py`.

## Input handling (`input.py`)

- `InputReader` runs in a daemon thread. It sets cbreak mode, reads raw bytes via `os.read(fd, 1024)` + `select`, and feeds them to `Vt100Parser`.
- `InputState` is the shared state between the reader thread and the main loop. The main loop reads `text`, `cursor`, `completions` for rendering. The reader writes them from `_on_key`.
- `Buffer(multiline=True)` manages editing state. `InputState.text` and `.cursor` are properties that delegate to `buffer.document`.
- Enter submits. Alt+Enter inserts a newline. Up/Down navigate within multiline text first, cycle history only at document boundaries.
- History lives on `InputState.history` (shared between reader and UI). `append_history` writes to disk on every call (crash-safe). Auto-dispatched commands (startup `/review`) also go through `append_history`. Up/Down do prefix search — the text at first keypress locks the prefix; only matching entries are shown.
- Tab sets `tab_pressed` flag. The main loop in `tui.py` handles completion (needs context like Engine access).

## Shell completion

- Completion is triggered by Tab on `!commands`. The pipeline lives in `input.py` as pure functions. The UI calls `query_shell_completions()` from a background thread.
- **Three tiers**, each falling through on empty results:
  1. **`complete_bash`** — sources bash-completion scripts in a non-interactive `bash -c` subprocess. Handles subcommands, flags, branch names, etc. ~80ms per call.
  2. **`complete_path`** — deterministic filesystem listing. Handles file/directory arguments.
  3. **`complete_executables`** — searches `PATH` for matching executables. Handles command names (first word only).
- `complete_bash` searches well-known directories for completion scripts (Homebrew, apt, Nix, MacPorts, Snap, `$BASH_COMPLETION_USER_DIR`). If the bash-completion framework is installed, it sources that first (handles lazy loading). Otherwise it sources per-command files directly.
- No pty-based completion. No terminal output parsing. All completion output is structured (one value per line from `COMPREPLY`).
- `_apply_completions` in the UI handles presentation: single match → auto-accept with suffix, multiple matches → extend common prefix and show menu.
- Completion helpers (`replace_shell_word`, `completion_suffix`, `shell_context_word`) are pure functions in `input.py`.

## Styling

- All styling lives in `styles.py` via a `rich.theme.Theme` object (`THEME`).
- Palette is Ayu Mirage. All hex values stay inside the `THEME` definition — nowhere else.
- Theme keys use `rbtr.*` namespace (e.g. `"rbtr.prompt"`, `"rbtr.bg.input"`).
- Plain-string constants (e.g. `PROMPT`, `ERROR`, `STYLE_DIM`) export the theme key names so Engine code can reference styles without importing Rich.
- Console is created with `Console(theme=THEME)`. Markup uses theme keys: `[rbtr.warning]...[/rbtr.warning]`.
- To add a new visual element: add a key to `THEME`, export a string constant, use it.

## Engine / UI boundary

- Engine lives in `engine.py`. UI lives in `tui.py`. They communicate through `queue.Queue[Event]`.
- Engine never imports Rich or touches the display.
- UI never runs commands or does I/O.
- `input.py` is a utility module — no Rich, no Engine. Pure input handling + completion functions. Safe to call from any thread.
- New output types → new event model in `events.py`, handle in both Engine and UI.

## Shell execution (`!commands`)

- Child processes get a **pty slave as stdin** (`os.openpty`). The reader thread has exclusive access to the real stdin — the child must never share it, or it races for keystrokes (e.g. steals Ctrl+C). The pty satisfies terminal-aware commands (`watch`, `less`) that need `ioctl`/`isatty` on stdin. `/dev/null` and `subprocess.PIPE` break those commands.
- `start_new_session=True` on Popen. The child gets its own session and process group. Save the PGID immediately after creation (`os.getpgid(proc.pid)`) — the PID may be gone by the time we need to kill the group.
- `communicate(timeout=0.02)` in a loop, checking `_check_cancel()` on each timeout. Never use a blocking `communicate()` or `proc.wait()` — both prevent cancel checks and can deadlock on full pipe buffers.

## Cancellation

- `_out()` calls `_check_cancel()` — every output emission is an automatic cancel checkpoint. Individual handlers only need explicit `_check_cancel()` between blocking operations that don't emit output (e.g. between API calls).
- `on_cancel` callback fires directly from the reader thread on Ctrl+C — no round-trip through the main loop. Sends SIGTERM to the saved PGID immediately.
- The `_TaskCancelled` cleanup sends SIGKILL via the saved PGID (not `os.getpgid(proc.pid)` — the process may already be dead). This handles children that trap SIGTERM (`watch`, `less`).
- Terminal setup disables ISIG in a single atomic `tcsetattr` call. A SIGINT handler is installed as a safety net — treats stray SIGINT as a cancel request instead of crashing with KeyboardInterrupt.
- "Cancelling…" appears in the active panel on `cancel_requested`. `TaskFinished(cancelled=True)` replaces it with "Cancelled." (or adds it fresh if the cancel was faster than the polling cycle).

## Threading

- Never block the main thread with I/O.
- Tasks run in daemon threads via `_start_task()`.
- Shell completion runs in short-lived daemon threads from the UI main loop (not Engine tasks — no events emitted).

## Input routing

- Plain text → LLM. `/command` → internal handler. `!command` → shell.
- Commands go in `_COMMANDS` dict with a handler in `_handle_command`.
- Keep commands minimal — if it could be an LLM question, it should be.
- Every submission is echoed as an Input panel before the response task starts.

## Code organization

- Constants in `constants.py`. No magic strings or numbers.
- Input handling in `input.py`. `InputState`, `InputReader`, and all completion functions.
- GitHub code in `src/rbtr/github/`. Auth in `auth.py`, API in `client.py`.
- Client functions take a `Github` instance. Don't create clients internally.
- Auth functions are public primitives callable by any consumer.

## HTTP

- Use httpx for direct HTTP. Never urllib or requests.
- Respect polling intervals. Cap pagination (`MAX_BRANCHES`).
- Don't add extra API calls to "verify" things PyGithub already handles.

## Errors

- Raise `RbtrError`. No `sys.exit()` in library code.

## CLI

- No click, argparse, typer. Plain `sys.argv` parsing with `print()` for `--help`.

## Dependencies

- Only add dependencies when their consuming code exists and is imported.

## Testing

- Default: test via Engine event contract. Create Engine with mock Session, call `run_task()`, assert on event sequence.
- Unit tests acceptable for pure functions with complex logic.
- Completion tests: test each tier independently (mock the others), test the orchestrator for fallthrough logic, test UI integration for threading/staleness.
- Never mock Engine internals to test other Engine methods.
- **pytest-mock only.** Use the `mocker` fixture for all mocking. Never import from `unittest.mock` — no `patch`, `MagicMock`, or `Mock` from there. Use `mocker.patch(...)`, `mocker.MagicMock()`, `mocker.patch.object(...)`, `mocker.patch.dict(...)`.
- **Flat functions, not classes.** Tests are top-level `def test_*()` functions. No `class Test*` grouping. Use descriptive function names with a shared prefix for related tests (e.g. `test_shell_*`, `test_login_*`).
- **Fixtures over setup.** Extract shared setup into `@pytest.fixture` functions. Use `tmp_path`, `monkeypatch`, `mocker` from pytest. No `setUp`/`tearDown`.
- **No parentheses on `@pytest.fixture`.** Write `@pytest.fixture`, not `@pytest.fixture()`.
- **Helper functions are fine.** Module-level pure helpers like `_make_engine()`, `_drain()`, `_inp()` are acceptable — not everything needs to be a fixture. Use fixtures when teardown or pytest integration (e.g. `mocker`, `tmp_path`) is needed.

## Types

- Name complex annotations. Use `type` aliases for repeated compound types like `list[tuple[str, str]]` or `dict[str, list[int]]`. The alias name carries meaning the structure doesn't.
- If a `dict` has a known fixed set of keys, use `TypedDict` (or Pydantic `BaseModel` if validation is needed). Don't pass around `dict[str, str]` when the shape is well-defined.
- No `NamedTuple`. Use `dataclass` for structured records or `TypedDict` for dict-shaped data. `NamedTuple` looks like a class but behaves like a tuple — it's iterable, indexable, destructurable by position, and compares equal to raw tuples with the same values. This causes subtle bugs and defeats the point of naming fields.
- When the same inline annotation appears more than once, that's a signal it should be an alias. One definition, one place to update.
- Keep aliases near the types they describe: module-level in the file that owns the concept, imported by consumers.

## Suppressions (`# noqa`, `# type: ignore`)

- Use sparingly and narrowly. Always specify the exact code (`# noqa: S602`, `# type: ignore[no-any-return]`) — never bare `# noqa` or `# type: ignore`.
- Before adding one, try to fix the root cause: add a proper type annotation, restructure the call, or import the right type. If the issue is in a third-party stub or stdlib typing gap, a suppression is acceptable.
- Each suppression must be obviously justified by reading the line alone. If it needs a comment explaining *why*, the code should probably be restructured instead.

## General

- No hacks. If it needs a comment explaining why it's weird, fix the root cause.
- Match existing style. Don't "improve" adjacent code.
- Clean up only orphans your changes created.

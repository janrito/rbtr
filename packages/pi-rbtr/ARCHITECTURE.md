# Architecture

Technical reference for contributors and maintainers.
For usage, see [README.md](README.md).

---

## What pi-rbtr is

pi-rbtr connects [pi][pi-pkg] to rbtr's code index. It
registers tools the model calls, adds what the index knows
to the results of the model's own `bash` and `read` calls,
and shows the index's state in pi's footer.

[pi-pkg]: https://pi.dev/docs/latest/packages

It does no indexing. rbtr's daemon does the work, and
pi-rbtr asks it over ZMQ; when no daemon answers, it runs
the `rbtr` CLI instead. Both reply with the same JSON
objects, defined in Python and generated into TypeScript
(`generated/`). pi-rbtr keeps no index or build state of its
own, only how to reach rbtr.

---

## How a session works

When a session starts, pi-rbtr finds rbtr, starts or
restarts its daemon if its version needs it, and reads the
index status. If the repository has no index, it starts
one. All of this runs in the background, and tools wait for
it.

A tool call goes to the daemon first, and to the CLI when
the daemon cannot be reached. Whatever rbtr replies goes
back to the model as rbtr's JSON object, errors included.

The daemon publishes notifications about its builds.
pi-rbtr subscribes to them and paints the footer: progress
while a build runs, the symbol count when it ends. A health
check every 30 s notices a daemon that stops or comes back.
A daemon that comes back without this repository, as one
restarted on a replaced index does, gets the session-start
decision again: with `autoIndex` on, pi-rbtr indexes it.

After the model's `bash` searches and its reads of long
files, pi-rbtr asks the daemon what the index knows and
appends it to the result.

---

## Choices

### Start in the background

pi holds the user's first prompt until every
`session_start` handler has finished, and finding and
starting rbtr can take seconds. So startup runs as a
background task, as pi's own MCP extension connects its
servers. Tools, and the notes on `bash` and `read` results,
wait for the task, and an abort ends the wait.

Each `rbtr` process costs most of a second in Python
imports. Startup therefore finds rbtr by asking each
candidate (`python3 -m rbtr`, `rbtr`, `uvx rbtr`) for its
daemon's status: the first answer says both that rbtr runs
and where its daemon is. Startup runs one process when a
daemon is running, and two when it has to start one.

When no candidate answers, pi-rbtr cannot tell why: pi
reports a program that cannot start as a silent exit 1. It
suggests installing rbtr, the likeliest cause, and does
nothing more.

The daemon is restarted only when it is older than the
extension. Versions are compared by their parts, and one
that cannot be read leaves the daemon running.

### Daemon first, the CLI as fallback

The daemon keeps the index open and its models loaded, so a
call costs milliseconds, where a CLI call pays for a new
process every time. The CLI keeps the tools working when no
daemon runs.

The daemon's sockets sit at fixed paths, so a restarted
daemon is reached again without looking for it. ZMQ never
fails a request sent to a socket nobody listens on, so
pi-rbtr checks the daemon's pid before each request: a
request to a daemon that has stopped would otherwise wait
out the whole time limit. The health check uses the same
pid check, and runs the CLI only while the daemon is gone.

zeromq's socket monitor could report a daemon's death at
once, without polling. It is untested on `ipc://`, and the
pid check needs nothing new; it is worth revisiting if the
30 s delay matters.

### Patient time limits

A daemon that is indexing answers a search late, after the
work in front of it, and asking again only adds to that
queue. So a tool call waits 120 s for the daemon, and the
CLI fallback 150 s, enough to outlast the client it starts.
An abort from the user ends any wait at once and is never
retried.

A read whose ref rbtr is building, or will build without
being asked, waits for that build to end, at most 30 s,
showing the build's phase in the tool's row, and then asks
again. It waits for the build only, not for embeddings,
which reads do not need. After 30 s the model gets rbtr's
"in progress" reply and can go on with other tools.

### Raw JSON replies

The model gets rbtr's JSON object as it is. Models parse
JSON well and take the fields they need; reformatting would
lose information or add size. A codemode script gets the
same object as `structuredContent`, and the renderers draw
the human-readable view from it.

Prose replies, such as an empty result echoing the call's
arguments, do not work beside codemode: a script's
`JSON.parse` fails on them, and in a trial on pi 0.99.1 the
model spent a turn wrapping the call in `try`/`catch`. A
read instead reports what rbtr resolved, such as the commit
it read, which the model cannot know otherwise.

Each tool's output schema lists rbtr's error object beside
its success reply. Leaving the error out would shorten
codemode's declaration of the tools by about a quarter
(4,466 to 3,446 characters), but codemode hands a script
the error object, so its declared type has to include it.

### What the model reads first

Open models reach for the tools they were trained on,
mostly `bash`: with the shipped text, `deepseek-v4-flash`
made 1.8% of its tool calls through rbtr. Leading each
description with what the tool replaces, usually grep,
tripled that (measured in `rbtr-agent-eval`'s screen of tool
use). pi lists tools in the order they are registered, so
the five tools that find code come first, and each keeps at
most two guidelines.

`rbtr_status` and `rbtr_gc` start inactive, so the prompt
the model reads first is about finding code; the loader,
`rbtr_index_tools`, activates them on request. pi restores
the tools a session loaded when it forks or resumes it, so
they stay active there.

The line saying the repository has an index is pi's `rbtr`
prompt section. Returning a whole `systemPrompt` instead
would make pi send it as an opaque prompt on every request,
and override the sections other extensions set.

Every tool declares the `rbtr` namespace and MCP-style
annotations, which codemode and permission extensions read.
`rbtr_gc` is the one destructive tool: it deletes only after
the user confirms what a dry run would drop, refuses
without a UI, and, like the loader, is not callable from a
codemode script.

Some providers send a list argument as a JSON-encoded
string. Each tool with a list parameter repairs that before
pi validates the call (`prepareArguments`), as pi's own
`edit` tool does.

### Facts in the agent's own tools

Leading with what each tool replaces still left most of an
open model's calls in `bash`. So pi-rbtr meets the model
there: after a `grep`, `rg`, `ag` or `git grep`, and after a
read of a long file, it appends what the index knows, at
most ten lines and nothing the output already shows. Every
line costs tokens on every call.

The facts come from the daemon only, because a CLI fallback
would start a process for every `bash` and `read`. A failure
adds nothing, so the tool's own output is never lost. A
search that looks only outside the repository gets nothing:
the index knows nothing true about another tree.

A file is outlined once while the outline is in the model's
context. pi-rbtr reads that from the session's context
entries, which follow `/tree` and compaction, rather than
keeping a set of its own, which would not. Within one turn
the entries cannot answer: pi runs a turn's calls in
parallel, and neither of two reads of a file has its result
in the context when the other's is noted, so both would be
outlined. pi-rbtr keeps
the files this turn's reads outline, and empties the set
when the next turn starts, by which time the results are
entries.

### The repository root comes from rbtr

rbtr names a repository by its git root's real path, which
differs from pi's working directory when pi runs in a
subdirectory or through a symlink. pi-rbtr takes the root
from rbtr's replies: a status reply states it even when
nothing is indexed. Notifications are matched to the
session by that root, and the notes name files from it.

### No build state in the extension

The daemon schedules builds and removes duplicates, so
pi-rbtr holds no promise or "already running" flag; the
footer follows the daemon's notifications. Which watched
refs are stale is rbtr's to decide too, so removing them is
one request rather than a second definition in TypeScript.

### A status line, not a widget

The index's state is one line, and `ctx.ui.setStatus` is
pi's primitive for one line of status. A widget would take
rows above or below the editor, and `setFooter` would
replace pi's whole footer, other extensions' statuses
included.

The footer animates only in pi's terminal UI. In RPC mode
every status is a message on stdout, so it paints one
static status per state. Without a UI it paints nothing,
because pi's theme may not exist there.

### Project settings only when trusted

The `command` setting is run at session start, so an
untrusted checkout must not choose it. The project's
`.pi/rbtr-index.json` is read only when pi trusts the
project, as pi treats its own `.pi/` files.
`/rbtr-settings` writes only that project file: project
settings are visible and versioned, and global ones are
edited by hand.

---

## Testing

Tests run the extension in a real pi session, with pi-ai's
faux model making the tool calls, against a real rbtr
daemon and a real git repository. The daemon runs with
embedding off (`RBTR_EMBED=false`, `tests/rbtr-env.ts`):
the extension needs no embeddings, and no model runs.
`tests/session.fixtures.ts` and `tests/global-setup.ts`
describe the setup.

Tests run on the installed pi, the one that loads the
extension: `vitest.config.ts` asks `pi` for its package and
resolves pi's packages and `typebox` from it, and global
setup stops a run that loaded another. CI installs the
latest pi from npm. `tsc` checks against the workspace's
locked pi. The locked pi does not work as the tests' pi: the
extension accepts any pi, and a user runs the pi they have
installed, so tests on the locked one check a pi nobody runs
(1.0.0, with 1.1.0 installed).

The repository is a clone of this checkout, so the tests
run over real code: global setup indexes it once and picks
what the tests use, a function used in another file and a
long file, by rules applied to rbtr's own replies. A test
names none of it, and takes each expected value from the
daemon's reply to the same request; a note's, from the facts
`annotate.ts` gives for the daemon's replies. What a note
says is tested in tables over a stand-in index
(`annotate.test.ts`), with the notes written out. An
invented sample application does not do this: its code and
its tests' names are written for each other, and say
nothing about real code.
Indexing the checkout in place does not work: CI's
depth-1 checkout has no history to compare, and tests
write branches and settings into their repository.

Tests read that repository and never write to it. A test
that writes settings or changes the watch set gets a small
repository of its own, made from two of the clone's files
(`writableRepo`); one that watches a build adds copies of a
long file, so the build lasts over a second (`longBuild`).
A copy of the whole clone does not work: each copy is a
build of the whole snapshot, 2.2 s alone at load average 6,
and several at once took tests past their 20 s waits at
load average 10 to 15.

The clone costs time and ties the tests to the checkout.
Global setup takes about 18 s and the whole run about 45 s,
at load average 7. A commit to the checkout can change what
setup picks, and so which code a test runs over: a test
that fails after an unrelated commit may have a new
subject. Setup asks the daemon through the extension's own
client, so a defect there fails the run in setup, before
the client's tests run.

The run shares one daemon. A test that stops, starts or
freezes the daemon gets its own (`ownDaemon`). A daemon per
test file does not work: every file's first test paid for
starting a daemon and indexing (up to 10 s at load average
9), past the tests' waits under load.

Tests check only what pi, the model or the user would see:
tool results, notifications, footer statuses, and the
processes the extension starts.

Conditions that are hard to reach use real processes:
stand-in scripts in `tests/fixtures/` for a missing or
stuck rbtr, and `SIGSTOP` for a silent daemon. `FakeDaemon`
only sends `index_error`, which a real daemon cannot be
made to send.

Pure functions with many edge cases keep table tests:
`args`, `classify`, `render`, `daemon-reconcile` and
`annotate`.

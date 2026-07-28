# rbtr

> Language-agnostic structural code intelligence for humans
> and clankers.

rbtr reads a codebase the way a reviewer does — as functions,
classes, methods, variables, and imports tied together by a
dependency graph — rather than as a flat wall of text. It builds
that structure into an index you can interrogate by name, by
snippet, or by meaning, and serves it from a background daemon
that keeps every watched ref current. The same index answers
questions for a person at a terminal and for an agent navigating
code through tools — in any language [tree-sitter] can parse.

[tree-sitter]: https://tree-sitter.github.io/tree-sitter/

## Why

Most questions about code are structural. *Who calls this?
What is in this file? What did this branch change? Where is
the retry logic?* The first three are graph and outline
queries. The last is a question about meaning, and the code
that answers it may well say `backoff` and never say `retry`.
Text search reaches all four indirectly at best.

For an agent the cost is sharper. Finding one function by
grepping means reading whole files into a context window that
has to hold everything else too. rbtr returns the symbol: its
name, its line range, and its source.

**Use grep instead** when you want an exact string — an error
message, a config key, a regex, every occurrence of an
identifier including the ones in comments. rbtr is structural,
not textual, and it will not beat `rg` at that. **Use an LSP**
for type-accurate rename and refactor across a project. rbtr
finds and reads; it does not edit.

## Demos

### Search & retrieval

Three ways to find code — by name, by snippet, by meaning —
and one score, fused from name, keyword and semantic channels.

![Search & retrieval](demo/output/index-search.gif)

### Structural navigation

A file's outline, the symbols that reference a symbol, and a
diff between two refs reported as changed *definitions* rather
than changed lines.

![Structural navigation](demo/output/structural-nav.gif)

### Agent integration

A pi session where the LLM answers questions about a codebase
by calling rbtr tools directly.

![Agent integration](demo/output/agent-integration.gif)

## Install

```bash
uv tool install "rbtr[all]"   # the code index CLI, every language
pi install npm:@rbtr/pi       # the pi extension
```

Plain `uv tool install rbtr` gives you eight languages — the
ones that ship as required dependencies. Everything else is an
extra, so `rbtr[all]` is the one to want unless you are keeping
the install small. See [Languages](#languages) for the split.

```bash
cd /path/to/your/repo
rbtr index                    # build the index
rbtr search "retry logic"     # search it
```

## How it fits together

**[`rbtr`](packages/rbtr/)** is the Python CLI and the daemon
behind it: it builds the index, keeps every watched ref
current, and answers queries. Everything else in this repo
integrates with it.

**[`pi-rbtr`](packages/pi-rbtr/)** is a thin TypeScript
extension that surfaces the same queries to an LLM as tools. It
talks to the daemon over ZMQ — a REP socket for
request/response, a PUB socket for progress, ready and error
notifications — and falls back to running the CLI when no
daemon is up.

**Languages are separate distributions.** Core ships no
grammars of its own. Each language is an `rbtr-lang-*` package
that registers through the `rbtr.languages` entry point, so
installing the package is all it takes to teach rbtr a
language — including one you publish yourself. Eight are
required dependencies of `rbtr`; the rest are extras.
[Writing a language plugin](packages/rbtr/README.md#writing-a-language-plugin)
is the guide.

**[`review-github-pr`](skills/review-github-pr/)** is a skill
that drives GitHub PR reviews through `gh api graphql`, and
**[`rbtr-eval`](packages/rbtr-eval/)** is an internal DVC
pipeline that measures and tunes search quality. Neither is
shipped with the CLI.

## Documentation

| Document                                                   | What's in it                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [`rbtr` README](packages/rbtr/README.md)                   | CLI reference, configuration, output modes, and how to write a language plugin             |
| [`rbtr` ARCHITECTURE](packages/rbtr/ARCHITECTURE.md)       | Indexing pipeline, data model, daemon, search fusion, garbage collection, design rationale |
| [`pi-rbtr` README](packages/pi-rbtr/README.md)             | The eight agent tools, slash commands, and extension settings                              |
| [`pi-rbtr` ARCHITECTURE](packages/pi-rbtr/ARCHITECTURE.md) | Session lifecycle, daemon-first transport with CLI fallback, rendering                     |
| [`rbtr-eval` README](packages/rbtr-eval/README.md)         | Benchmark and tuning pipeline, and what the measurements found                             |
| [`demo` README](demo/README.md)                            | Recording the terminal demos above with VHS                                                |

### Languages

Each language is its own package, with a README describing
exactly what it extracts.

| Language                    | Package                                                                         | Install                   |
| --------------------------- | ------------------------------------------------------------------------------- | ------------------------- |
| Bash / shell                | [`rbtr-lang-bash`](packages/rbtr-lang-bash/README.md)                           | default                   |
| C                           | [`rbtr-lang-c`](packages/rbtr-lang-c/README.md)                                 | default                   |
| C++                         | [`rbtr-lang-cpp`](packages/rbtr-lang-cpp/README.md)                             | default                   |
| CSS                         | [`rbtr-lang-css`](packages/rbtr-lang-css/README.md)                             | `rbtr[css]`               |
| Go                          | [`rbtr-lang-go`](packages/rbtr-lang-go/README.md)                               | `rbtr[go]`                |
| HCL / Terraform             | [`rbtr-lang-hcl`](packages/rbtr-lang-hcl/README.md)                             | `rbtr[hcl]`               |
| HTML                        | [`rbtr-lang-html`](packages/rbtr-lang-html/README.md)                           | default                   |
| Java                        | [`rbtr-lang-java`](packages/rbtr-lang-java/README.md)                           | `rbtr[java]`              |
| JavaScript, TypeScript, TSX | [`rbtr-lang-javascript`](packages/rbtr-lang-javascript/README.md)               | default                   |
| JSON                        | [`rbtr-lang-json`](packages/rbtr-lang-json/README.md)                           | `rbtr[json]`              |
| Less                        | [`rbtr-lang-less`](packages/rbtr-lang-less/README.md)                           | `rbtr[less]`              |
| Markdown                    | [`rbtr-lang-markdown`](packages/rbtr-lang-markdown/README.md)                   | default                   |
| Python                      | [`rbtr-lang-python`](packages/rbtr-lang-python/README.md)                       | default                   |
| reStructuredText            | [`rbtr-lang-rst`](packages/rbtr-lang-rst/README.md)                             | default                   |
| Ruby                        | [`rbtr-lang-ruby`](packages/rbtr-lang-ruby/README.md)                           | `rbtr[ruby]`              |
| Rust                        | [`rbtr-lang-rust`](packages/rbtr-lang-rust/README.md)                           | `rbtr[rust]`              |
| SCSS                        | [`rbtr-lang-scss`](packages/rbtr-lang-scss/README.md)                           | `rbtr[scss]`              |
| SQL                         | [`rbtr-lang-sql`](packages/rbtr-lang-sql/README.md)                             | `rbtr[sql]`               |
| Svelte                      | [`rbtr-lang-svelte`](packages/rbtr-lang-svelte/README.md)                       | `rbtr[svelte]`            |
| TOML                        | [`rbtr-lang-toml`](packages/rbtr-lang-toml/README.md)                           | `rbtr[toml]`              |
| tree-sitter queries         | [`rbtr-lang-tree-sitter-query`](packages/rbtr-lang-tree-sitter-query/README.md) | `rbtr[tree-sitter-query]` |
| Vue                         | [`rbtr-lang-vue`](packages/rbtr-lang-vue/README.md)                             | `rbtr[vue]`               |
| YAML                        | [`rbtr-lang-yaml`](packages/rbtr-lang-yaml/README.md)                           | `rbtr[yaml]`              |

A file in a language with no plugin installed is still
indexed — it falls back to line-based chunking, so it stays
searchable, just without structure.

## Development

```bash
just setup                  # uv sync + bun install
just check                  # lint, typecheck, and every test suite
just fmt                    # auto-fix (Python, TypeScript, SQL, Markdown)
```

`just --list` has the rest. `just check` needs network access:
it validates the review skill's GraphQL queries against
GitHub's published schema. `just lint` runs import-linter over
the layer boundaries, so a misplaced import fails the build.

Agent conventions live in [`.agents/skills/`](.agents/skills/) —
data handling, testing, and language-plugin authoring.

## Troubleshooting

| Symptom                                                                | Recovery                                                                                                                                                                                            |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rbtr CLI not found` or `daemon start/restart failed` at session start | Run `rbtr daemon status`; if it's down, `rbtr daemon start`. Concurrent sessions converge on one daemon, so this is usually transient — a busy index reports "temporarily unavailable" and retries. |
| `Index database is locked by another process`                          | The running daemon holds the index lock; route commands through it (`rbtr daemon status`). Only `rbtr daemon stop` a genuinely stale daemon — never kill a healthy one.                             |
| `No index found` for a repo that should be indexed                     | Run `rbtr index` (or `/rbtr-index` in pi); confirm with `rbtr status`.                                                                                                                              |
| The daemon refuses to start, naming languages it cannot load           | The index holds chunks from a plugin this install is missing. Install it, or start with `rbtr index --allow-missing-plugins` to proceed without it.                                                 |

`rbtr config` prints the resolved paths (including the daemon
log) and the language plugins actually loaded.

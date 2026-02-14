# Architecture

Permanent design decisions for rbtr. These are not up for renegotiation on a per-feature basis.

## 1. One Implementation Per Operation

Every operation (read a file, search symbols, query the graph, etc.) exists as a single plain function. The CLI calls it. The PydanticAI agent calls it. There are no wrappers, adapters, or parallel implementations. If the function signature needs to change, it changes in one place.

## 2. The LLM Is an Assistant, Not an Actor

The LLM never posts comments, approves PRs, or takes actions on GitHub. It answers questions and provides analysis. The human makes all decisions. The LLM's outputs are always presented to the user, never forwarded anywhere automatically.

## 3. All Storage Is Local and Embedded

No external services for data storage. The vector index (LanceDB) and knowledge graph (Kùzu) are embedded databases stored on disk alongside the repo. No Docker, no servers, no accounts. The tool works offline after the initial GitHub API fetch.

## 4. Everything Is Typed

All code is strictly typed and checked by mypy in strict mode. Pydantic models for all data boundaries (GitHub API responses, index records, graph query results). No `Any` unless forced by a third-party library.

## 5. Configuration Lives in the Repo

Each repo has a `.rbtr.toml` at the root for configuration and a `.rbtr/` directory for cached data (vector index, graph database, API responses). Configuration is loaded via pydantic-settings with TOML as the primary source, environment variables as fallback. The `.rbtr/` directory should be gitignored.

## 6. Caching by Ref

Both the vector index and the knowledge graph are cached on disk keyed by repository + commit SHA. Re-running against the same ref is instant. Changed files are re-indexed incrementally — we never rebuild from scratch unless the cache is deleted.

## 7. Tree-sitter for Structural Analysis

Structural code analysis (call graphs, inheritance, symbol extraction for the knowledge graph) uses tree-sitter. Regex is acceptable for simple text search (grep-like operations in repo.py) but not for anything that needs to understand code structure. Python grammar first, other languages added per grammar.

## 8. Retrieval Has Two Layers

- **Semantic** (LanceDB): embedding-based similarity search. Answers "what code is related to this concept?"
- **Structural** (Kùzu): graph traversal over parsed relationships. Answers "what calls this function?", "what does this module depend on?", "what is the blast radius of this change?"

These are complementary, not redundant. Both are available to the user and the agent as shared operations.

## 9. The GitHub API Is the Only Network Dependency

pygit2 operates on the local clone. LanceDB and Kùzu are embedded. The only thing that requires network access is fetching PR data from the GitHub REST API (and embedding generation if using an API-based provider). Once fetched, the PR data is held in memory for the session.

## 10. The Primary Interface Is a TUI

The main experience is an interactive TUI built with Textual (from the Rich ecosystem). You navigate commits, view diffs, ask the LLM questions, and explore the codebase all within the terminal application. Some operations may also be exposed as standalone CLI commands for scripting, but the TUI is the primary flow — not an afterthought.

## 11. Dependencies Are Intentional

Every dependency earns its place:

| Dependency  | Reason                                                             |
| ----------- | ------------------------------------------------------------------ |
| pydantic-ai | Agent framework with structured output and tool registration       |
| pygit2      | Read repo contents at arbitrary refs without checkout, via libgit2 |
| PyGithub    | GitHub API client — handles auth, pagination, rate limiting        |
| rich        | Terminal rendering: syntax highlighting, tables, panels, progress  |
| textual     | TUI framework for the main interactive interface                   |
| click       | CLI argument parsing and entry points                              |
| lancedb     | Embedded vector database for semantic search                       |
| kuzu        | Embedded graph database for structural knowledge graph             |
| tree-sitter | Code parsing for structural analysis                               |

New dependencies require justification. "It would be convenient" is not sufficient.

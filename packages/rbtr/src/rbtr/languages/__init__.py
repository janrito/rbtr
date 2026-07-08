"""Language plugins: the `LanguageRegistration` contract and its manager.

A plugin is a `LanguageRegistration` value registered via the `rbtr.languages`
entry-point group. The pieces live in named submodules — import from them
directly:

- `rbtr.languages.registration` — the authoring surface (`LanguageRegistration`,
  `ModuleStyle`, the resolver aliases, `load_query`, capture/import helpers).
- `rbtr.languages.manager` — `get_manager()` discovers registered plugins and
  drives detection and grammar loading.
- `rbtr.languages._resolvers` — the engine's built-in resolvers (private).
"""

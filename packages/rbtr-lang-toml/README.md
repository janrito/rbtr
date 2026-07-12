# rbtr-lang-toml

TOML support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[toml]`.

## What it ingests

Tables and array-tables become **config-key** chunks (TOML is data, not code).
A dotted table splits into its last-segment name plus the preceding path as
scope.

## Chunks produced

```toml
[project]              # config_key "project"
[tool.ruff]            # config_key "ruff",   scope "tool"
[tool.ruff.lint]       # config_key "lint",   scope "tool::ruff"
[[locales]]            # config_key "locales"
```

## Embedded / injected chunks

None. TOML does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-toml` grammar. No dependency on other language plugins.

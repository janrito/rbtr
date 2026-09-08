# rbtr-lang-yaml

YAML support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[yaml]`.

## What it ingests

Top-level mapping keys become **config-key** chunks (YAML is data, not code).
Non-mapping YAML (a bare sequence or scalar) produces a single fallback chunk.

A `$ref` names another document, as OpenAPI splits a specification across
files, and becomes an **import**. So does a `uses` naming a path, which is how
a workflow reads an action kept in the same repository; a `uses` naming
anything else reads it from elsewhere and is left alone.

## Chunks produced

```yaml
name: CI          # config_key "name"
on: [push]        # config_key "on"
jobs:             # config_key "jobs"
  build: …
```

## Embedded / injected chunks

None. YAML does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-yaml` grammar. No dependency on other language plugins.

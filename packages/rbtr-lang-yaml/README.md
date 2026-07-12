# rbtr-lang-yaml

YAML support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[yaml]`.

## What it ingests

Top-level mapping keys become **config-key** chunks (YAML is data, not code).
Non-mapping YAML (a bare sequence or scalar) produces a single fallback chunk.

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

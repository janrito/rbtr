# rbtr-lang-json

JSON support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[json]`.

## What it ingests

Top-level keys of the root object become **config-key** chunks (JSON is data,
not code). Non-object JSON (arrays, scalars) produces no structural chunks and
falls through to plaintext.

## Chunks produced

```json
{
  "name": "my-project",     // config_key "name"
  "version": "1.0.0",       // config_key "version"
  "dependencies": { … }     // config_key "dependencies"
}
```

## Embedded / injected chunks

None. JSON does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-json` grammar. No dependency on other language plugins.

# rbtr-lang-json

JSON support for [rbtr]. Optional plugin — install with
`pip install rbtr[json]`.

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

Top-level keys of the root object become **config-key** chunks (JSON is data,
not code). Non-object JSON (arrays, scalars) produces no structural chunks and
falls through to plaintext.

A `$ref` or an `extends` key names another file, so its chunk is a config key that also
imports that file, linking the two documents. A `$ref`'s pointer — the part after the
`#` — names a key inside the file it reaches, so the edge lands on that key rather than
on the document as a whole. Keys are not scoped by their parents, so
`#/definitions/Named` reaches the first key named `Named`; a pointer alone (`#/z`) is
into the same document.

## Chunks produced

```json
{
  "name": "my-project",     // config_key "name"
  "version": "1.0.0",       // config_key "version"
  "dependencies": { … }     // config_key "dependencies"
}
```

```json
{
  "extends": "./base.json",          // config_key "extends", imports [{module: ./base.json}]
  "a": { "$ref": "b.json#/x" }       // config_key "a";
                                     // config_key "$ref", imports [{module: b.json, names: x}]
}
```

## Embedded / injected chunks

None. JSON does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-json` grammar. No dependency on other language plugins.

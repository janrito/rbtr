# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                         |
| ------------- | ----------------------------- |
| model         | `openai-chat:zai-org/GLM-5.2` |
| total queries | 3621                          |
| expanded      | 3621 / 3621 (100%)            |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 541  | 5.2          | 2.0          |
| concept    | 1566 | 5.3          | 2.0          |
| identifier | 1514 | 5.2          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 737   | 737      | 100% |
| astral-sh__uv      | 894   | 894      | 100% |
| badlogic__pi-mono  | 659   | 659      | 100% |
| django__django     | 769   | 769      | 100% |
| rbtr__rbtr         | 562   | 562      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 913   | 913      | 100% |
| concept    | 1526  | 1526     | 100% |
| docstring  | 389   | 389      | 100% |
| name       | 793   | 793      | 100% |

## Examples

### concept: `no-debug` (`astral-sh__uv`)

````toml
how to configure build settings to strip debug info and minimize binary size
````

- **keywords:** strip_debug_symbols, strip, -s, LDFLAGS, minimize_size, debug_info,
  strip_symbols
- **variants:** remove debug symbols from compiled binary to reduce size, compiler
  linker flags for stripping debug information and shrinking output

### concept: `extend-aliases` (`astral-sh__uv`)

````toml
how to configure custom import alias mappings for Home Assistant helper modules
````

- **keywords:** custom_components, import_alias, _import_hook, module_resolver,
  alias_mapping
- **variants:** set up custom module remapping for Home Assistant helper imports,
  register import aliases for HA utility modules

### concept: `PageBreak` (`anthropics__skills`)

````javascript
JavaScript library for generating Word documents with paragraphs tables headers and footers
````

- **keywords:** docx, Document, Paragraph, Table, Header, Footer, Packer
- **variants:** JS library to create .docx files with structured content like paragraphs
  and tables, programmatically build Word documents with headers footers and tables in
  JavaScript

### concept: `packageDirs` (`badlogic__pi-mono`)

````javascript
how to get all subdirectory names from a directory in Node.js
````

- **keywords:** readdirSync, withFileTypes, isDirectory, dirent, fs.readdir
- **variants:** list only folders inside a path using fs, filter directory entries to
  get child folders in Node

### concept: `PYPI_PUBLISH_URL` (`astral-sh__uv`)

````rust
default PyPI upload URL for publishing packages
````

- **keywords:** upload_url, repository, pypirc, twine upload, distutils
- **variants:** where does twine send packages by default, how to set the PyPI
  repository URL for publishing

### identifier: `clippy` (`astral-sh__uv`)

````toml
workspace::lints::clippy
````

- **keywords:** clippy_lints, clippy_rules, lint_clippy, clippy_checks, ClippyLint
- **variants:** collect or register clippy lint rules in a workspace, define
  clippy-specific lint passes and check configurations

### identifier: `sourceMap` (`anthropics__skills`)

````json
sourceMap
````

- **keywords:** source_map, sourceMappingURL, dbg_file, map_file, orig_source_map
- **variants:** map transpiled or minified code back to original source code, debugging
  aid that links generated output to its original source positions

### identifier: `buildTree` (`badlogic__pi-mono`)

````javascript
/**
       * Build tree structure from flat entries.
````

- **keywords:** build_tree, construct_tree, flat_to_tree, build_hierarchy,
  build_tree_structure
- **variants:** convert a flat list of nodes into a nested parent-child tree, organize
  flat entries into a hierarchical tree based on parent references

### identifier: `PUBLISH_DELAY_SECS` (`astral-sh__uv`)

````python
# Delay between `cargo publish` calls to respect crates.io rate limits.
````

- **keywords:** publish_delay, publish_throttle, publish_interval, crates_rate_limit,
  publish_cooldown
- **variants:** throttle delay between publishing crates to crates.io to avoid hitting
  rate limits, wait period between successive crate publish commands for rate limiting

### identifier: `rootConst` (`django__django`)

````javascript
rootConst = "root"
````

- **keywords:** root_constant, ROOT_KEY, baseConst, rootNode, ROOT_DIR, topLevelConst,
  rootIdentifier
- **variants:** constant defining the root or top-level identifier in a hierarchy,
  hardcoded base path or namespace prefix used as the starting point in a tree structure

### code: `stream` (`anthropics__skills`)

````typescript
stream = client.messages.stream({
  model: "claude-opus-4-7",
  max_tokens: 64000,
  messages: [{ role: "user", content: "Hello" }],
})
````

- **keywords:** messages.stream, client, claude-opus, max_tokens, streaming
- **variants:** call Anthropic Claude messages API with streaming enabled, initiate a
  streamed chat completion request to Claude with a user message

### code: `get_hidden_node_count` (`django__django`)

````javascript
get_hidden_node_count(id) {
            const cache = SelectBox.cache[id] || [];
            return cache.filter((node) => node.displayed === 0).length;
        }
````

- **keywords:** get_hidden_node_count, SelectBox, cache, displayed, filter
- **variants:** count nodes in SelectBox cache where displayed is zero, return number of
  hidden nodes from cached SelectBox entries

### code: `--green` (`anthropics__skills`)

````css
--green: #788c5d;
````

- **keywords:** green, 788c5d, css_variable, color_token, palette
- **variants:** CSS custom property defining a green color hex value, --green: #788c5d
  color variable definition

### code: `key` (`astral-sh__uv`)

````python
def key(self) -> str:
        if self.variant:
            return f"{self.implementation}-{self.version}+{self.variant}-{self.triple.platform}-{self.triple.arch}-{self.triple.libc}"
        else:
````

- **keywords:** key, variant, implementation, version, triple, platform
- **variants:** build a unique identifier string from implementation, version, variant,
  and target triple components, construct a composite key combining package version,
  build variant, and platform/arch/libc tuple

### code: `ImportedModelBackend` (`django__django`)

````python
class ImportedModelBackend(ModelBackend):
    pass
````

- **keywords:** ImportedModelBackend, ModelBackend, authentication_backend, imported,
  django
- **variants:** empty subclass of Django ModelBackend for imported users, custom
  authentication backend that inherits ModelBackend without overriding methods

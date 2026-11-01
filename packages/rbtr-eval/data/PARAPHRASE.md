# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                                         |
| --------------- | --------------------------------------------- |
| model           | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| concept queries | 1564                                          |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 307 |
| astral-sh__uv      | 402 |
| badlogic__pi-mono  | 279 |
| django__django     | 332 |
| rbtr__rbtr         | 244 |

## Per language

| language   | n   |
| ---------- | --- |
| python     | 367 |
| typescript | 206 |
| javascript | 175 |
| bash       | 170 |
| css        | 145 |
| rust       | 121 |
| markdown   | 98  |
| json       | 97  |
| yaml       | 41  |
| rst        | 35  |
| sql        | 31  |
| toml       | 31  |
| plaintext  | 30  |
| html       | 17  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `make_id` (`django__django`)

````python
def make_id(target):
            """
            Simulate id() reuse for distinct senders with non-overlapping
            lifetimes that would require memory contention to reproduce.
            """
            if isinstance(target, Sender):
                return 0
            return _make_id(target)
````

> **concept:** generate a unique identifier for a sender object

### `merge_hashes` (`astral-sh__uv`)

````rust
/// Merge repeated hashes for a requirement or constraint into the hash map.
fn merge_hashes(
    hashes: &mut FxHashMap<VersionId, Vec<HashDigest>>,
    id: VersionId,
    incoming: Vec<HashDigest>,
    requirement: impl Display,
) -> Result<(), HashStrategyError> {
    if incoming.is_empty() {
        return Ok(());
    }

    if !matches!(&id, VersionId::ArchiveUrl { .. }) {
        hashes.insert(id, incoming);
        return Ok(());
    }

    if let Some(existing) = hashes.get_mut(&id) {
        return merge_digests(existing, &incoming, requirement);
    }

    let mut merged = Vec::new();
    merge_digests(&mut merged, &incoming, requirement)?;
    hashes.insert(id, merged);
    Ok(())
}
````

> **concept:** combine and store hash digests for a version identifier

### `_JS_QUERY` (`rbtr__rbtr`)

````python
# JS uses `identifier` for class names, TS uses `type_identifier`.
_JS_QUERY = (
    """\
(class_declaration
  name: (identifier) @_cls_name) @class

"""
    + _SHARED
)
````

> **concept:** how to detect class declaration names in both JavaScript and TypeScript
> syntax trees

### `` (`django__django`)

````javascript
// Call the preDispatch hook for the mapped type, and let it bail if desired
````

> **concept:** how to abort or prevent an action in the predispatch hook

### `` (`django__django`)

````bash
python manage.py migrate
````

> **concept:** how do I apply database schema changes to my app

### `ANTHROPIC_API_KEY` (`badlogic__pi-mono`)

````bash
# Option 1: Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
````

> **concept:** how to set the anthropic api key as an environment variable

### `UserSearchInput` (`anthropics__skills`)

````typescript
type UserSearchInput = z.infer<typeof UserSearchInputSchema>;
````

> **concept:** derive a TypeScript type from a Zod validation schema.

### `[0.67.5] - 2026-04-16` (`badlogic__pi-mono`)

````markdown
## [0.67.5] - 2026-04-16
````

> **concept:** what changed in the latest release

### `overrides` (`astral-sh__uv`)

````toml
[[profile.default.overrides]]
filter = 'package(uv-keyring)'
test-group = 'serial'


````

> **concept:** how to assign a specific test group for a package

### `` (`rbtr__rbtr`)

````bash
# keyword search
````

> **concept:** sing command line tools for searching files and text

# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                    |
| ------------- | ------------------------ |
| model         | `openai:zai-org/GLM-5.1` |
| total queries | 2586                     |
| expanded      | 2586 / 2586 (100%)       |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 402  | 5.1          | 2.0          |
| concept    | 1155 | 5.1          | 2.0          |
| identifier | 1029 | 5.1          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 224   | 224      | 100% |
| astral-sh__uv      | 626   | 626      | 100% |
| badlogic__pi-mono  | 505   | 505      | 100% |
| django__django     | 807   | 807      | 100% |
| rbtr__rbtr         | 424   | 424      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 625   | 625      | 100% |
| concept    | 1122  | 1122     | 100% |
| docstring  | 214   | 214      | 100% |
| name       | 625   | 625      | 100% |

## Examples

### concept: `node` (`badlogic__pi-mono`)

````json
how to set minimum runtime version requirement in package config
````

- **keywords:** engines, python_requires, requires-python, rust-version, engine-strict
- **variants:** specify compatible runtime version in package manifest, enforce minimum
  language version in package metadata

### concept: `Live Documentation Sources` (`anthropics__skills`)

````markdown
# Live Documentation Sources
````

- **keywords:** doc_sources, live_docs, documentation_provider, doc_generator,
  realtime_docs
- **variants:** where does live documentation pull content from, how are documentation
  sources loaded dynamically

### concept: `Footer class` (`rbtr__rbtr`)

````markdown
how does the footer status bar show animation during long-running operations
````

- **keywords:** progress_bar, status_bar_animation, spinner, loading_indicator,
  busy_indicator
- **variants:** how the status bar displays progress feedback for running tasks, what
  animation plays in the footer while operations are in progress

### concept: `homeassistant.components.rmvtransport` (`astral-sh__uv`)

````markdown
what Python package does the RMV transport integration depend on
````

- **keywords:** pyrmv, rmv_transport, manifest.json, requirements, DEPENDENCIES
- **variants:** which pip package is required by the RMV transport component, Python
  library dependency for RMV public transit integration

### concept: `makeRelParts` (`django__django`)

````javascript
split a relationship name into plugin ID and resource ID parts
````

- **keywords:** parse_relationship, split_identifier, plugin_resource, decompose_name,
  relationship_parser
- **variants:** how to parse relationship identifiers into plugin and resource
  components, extracting plugin and resource IDs from a compound relationship name

### identifier: `cursor_iter` (`django__django`)

````python
"""
    Yield blocks of rows from a cursor and ensure the cursor is closed when
    done.
````

- **keywords:** iter_chunks, fetchmany_batches, cursor_generator, chunked_rows,
  batch_fetch
- **variants:** yield rows in batches from a database cursor with cleanup, iterate over
  database cursor results in chunks and close cursor afterward

### identifier: `Fixed` (`badlogic__pi-mono`)

````markdown
Changelog > [0.52.7] - 2026-02-06.Fixed
````

- **keywords:** release_notes, version_fix, changelog_fix, patch_notes, bugfix_0_52_7
- **variants:** fixes in version 0.52.7 release, bug fixes listed in changelog for
  0.52.7

### identifier: `findVisibleAncestor` (`badlogic__pi-mono`)

````javascript
// Find nearest visible ancestor for a node
````

- **keywords:** nearest_visible_ancestor, closest_visible_parent, find_visible_ancestor,
  get_visible_parent, visible_node_ancestor
- **variants:** traverse up the DOM tree to find the first ancestor element that is not
  hidden, find closest displayed parent of a hidden DOM node

### identifier: `Graceful degradation` (`rbtr__rbtr`)

````markdown
## Graceful degradation
````

- **keywords:** fallback_handling, degrade_gracefully, graceful_fallback,
  resilience_pattern, failover_handling
- **variants:** continue operating with reduced functionality when components fail,
  design system to maintain partial operation during failures

### identifier: `launcher` (`astral-sh__uv`)

````rust
/// The name of the launcher shim.
````

- **keywords:** launcher_shim_name, bootstrap_launcher, launch_wrapper, launcher_stub,
  shim_identifier
- **variants:** name of the thin launcher wrapper used to start an application,
  identifier for the bootstrap shim that launches the main process

### code: `resolveCommand` (`rbtr__rbtr`)

````typescript
function resolveCommand(command: string): ResolvedCommand {
  const trimmed = command.trim();
````

- **keywords:** resolveCommand, ResolvedCommand, command, trimmed, trim
- **variants:** resolve a command string into a ResolvedCommand, parse and trim a
  command string before resolution

### code: `django/forms/jinja2/django/forms/errors/list/ul.html` (`django__django`)

````html
{% if errors %}<ul class="{{ error_class }}"{% if errors.field_id %} id="{{ errors.field_id }}_error"{% endif %}>{% for error in errors %}<li>{{ error }}</li>{% endfor %}</ul>{% endif %}
````

- **keywords:** error_class, errors.field_id, form_errors, error_list, field_id_error
- **variants:** Django template rendering form field errors as an unordered list with
  CSS class and ID, Jinja2 macro to display validation error messages in a ul element

### code: `publishConfig` (`rbtr__rbtr`)

````json
"publishConfig": {
    "access": "public"
  }
````

- **keywords:** publishConfig, access, public, npm_publish, scoped_package
- **variants:** npm package.json publishConfig access public for scoped package,
  configure npm scoped package for public registry access

### code: `isType` (`django__django`)

````javascript
function isType(value, type) {
  return Object.prototype.toString.call(value) === "[object ".concat(type, "]");
}
````

- **keywords:** isType, Object.prototype.toString.call, object type check, toString type
  detection, [object
- **variants:** check value type using Object.prototype.toString, type guard via
  toString call and [object] tag comparison

### code: `TimerPanel` (`badlogic__pi-mono`)

````typescript
class TimerPanel extends BaseOverlay {
	private seconds = 0;
````

- **keywords:** TimerPanel, BaseOverlay, seconds, timer_overlay, countdown_panel
- **variants:** timer overlay UI panel with seconds counter, class extending BaseOverlay
  to display a timer with elapsed seconds

# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                                         |
| ------------- | --------------------------------------------- |
| model         | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| total queries | 3623                                          |
| expanded      | 3623 / 3623 (100%)                            |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 582  | 5.2          | 1.9          |
| concept    | 1595 | 5.2          | 1.9          |
| identifier | 1446 | 5.0          | 1.9          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 735   | 735      | 100% |
| astral-sh__uv      | 904   | 904      | 100% |
| badlogic__pi-mono  | 651   | 651      | 100% |
| django__django     | 746   | 746      | 100% |
| rbtr__rbtr         | 587   | 587      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 967   | 967      | 100% |
| concept    | 1558  | 1558     | 100% |
| docstring  | 389   | 389      | 100% |
| name       | 709   | 709      | 100% |

## Examples

### concept: `test_invalid_type_end_row_range` (`django__django`)

````python
test that RowRange window frame rejects a non-integer end argument
````

- **keywords:** RowRange, window_frame, validate_end, type_error, assertRaises
- **variants:** test that RowRange window frame throws on a non-integer boundary, verify
  RowRange frame rejects invalid end value type

### concept: `current_ld` (`astral-sh__uv`)

````bash
# Make mold the default linker
````

- **keywords:** LINKER, ld, lld, gold, use_mold, CMAKE_LINKER
- **variants:** how to switch the compiler to use the mold linker by default, set the
  default linker to mold in the build toolchain

### concept: `DirtyWorktree` (`rbtr__rbtr`)

````python
represent a repository with uncommitted working tree changes and its current tree hash
````

- **keywords:** working_tree, dirty, head_hash, git_status, commit_sha
- **variants:** get the current commit with uncommitted modifications, determine repo
  state and hash when files are modified but not staged, serialize repo state including
  dirty working directory

### concept: `execute` (`badlogic__pi-mono`)

````typescript
delay processing and record execution order for a specific input value
````

- **keywords:** sleep, throttle, schedule_task, defer_result, ordered_registry,
  sequence_tracker
- **variants:** how to queue a job and log the order it runs for a given key, buffer a
  task until later and persist its run sequence per value

### concept: `RESY` (`badlogic__pi-mono`)

````bash
how to set the default render resolution with an environment variable override
````

- **keywords:** RENDER_RESOLUTION, set_display_size, default_resolution, env_override,
  configure_graphics
- **variants:** override the startup video output dimensions, control the initial screen
  width and height

### identifier: `--default-button-bg` (`django__django`)

````css
html[data-theme="light"],
:root::--default-button-bg
````

- **keywords:** css variable, theme variable, custom property, light-theme-color,
  button_default_bg
- **variants:** define a CSS custom property for the default button background color
  under the light color-scheme, set theme-scoped variable value inside an html
  data-theme selector

### identifier: `IndexErrorArticle` (`django__django`)

````python
IndexErrorArticle
````

- **keywords:** index_out_of_bounds, ArrayIndexOutOfBounds, IndexOutOfBoundsException,
  list_index_error, invalid_index
- **variants:** exception thrown when accessing an array or list with an out-of-range
  index, error raised when an index is outside the valid bounds of a collection

### identifier: `action` (`django__django`)

````python
"""
    Conveniently add attributes to an action function::
````

- **keywords:** action_decorator, decorate_action, attach_metadata,
  action_attribute_setter
- **variants:** attach metadata or extra attributes to an action function, wrap a
  function to set additional fields

### identifier: `.section-header` (`anthropics__skills`)

````css
.section-header
````

- **keywords:** section_title, heading_block, header_label, block_heading,
  section_title_bar
- **variants:** stylized title or label displayed at the top of a content section,
  heading element that introduces and labels a page block

### identifier: `getDownloadUri` (`anthropics__skills`)

````javascript
// ---- Util ----
````

- **keywords:** helper, utility, utils, common_functions, misc
- **variants:** shared helper functions used across the codebase, generic utility module
  with common operations

### code: `run_session` (`anthropics__skills`)

````python
def run_session(client, session_id: str):
    """Stream events and handle custom tool calls."""
    while True:
        with client.beta.sessions.stream(
            session_id=session_id,
        ) a
````

- **keywords:** run_session, session_id, beta.sessions.stream, stream_events,
  custom_tool_calls
- **variants:** stream session events and process custom tool calls, capture the beta
  sessions stream for the given session id

### code: `__dirname` (`anthropics__skills`)

````typescript
__dirname = dirname(fileURLToPath(import.meta.url))
````

- **keywords:** __dirname, dirname, fileURLToPath, import.meta.url, ESM_path
- **variants:** get the folder path of the current ES module, convert the module URL to
  a filesystem directory path

### code: `GitUrl` (`astral-sh__uv`)

````rust
impl std::fmt::Display for GitUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.url)
    }
}
````

- **keywords:** GitUrl, Display, Formatter, fmt, url
- **variants:** render the GitUrl by writing its underlying url string to the formatter

### code: `FakeReply` (`rbtr__rbtr`)

````typescript
type FakeReply = Response | ((request: Request) => Response);
````

- **keywords:** FakeReply, Response, Request, union_type, reply_middleware
- **variants:** a response or a function mapping request to response, type alias that is
  either a Response object or a handler that takes a Request and returns one

### code: `buildWorstExamples` (`badlogic__pi-mono`)

````javascript
function buildWorstExamples(records, top) {
	const scored = [...records].sort((a, b) => {
		const aScore = a.inflationRatio === null
````

- **keywords:** buildWorstExamples, inflationRatio, records, sort, topWorst
- **variants:** sort the worst records by their inflation ratio and return the top ones

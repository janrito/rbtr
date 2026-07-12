# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                         |
| ------------- | ----------------------------- |
| model         | `openai-chat:zai-org/GLM-5.2` |
| total queries | 2357                          |
| expanded      | 2357 / 2357 (100%)            |

## Per-kind breakdown

| query_kind | n   | avg_keywords | avg_variants |
| ---------- | --- | ------------ | ------------ |
| code       | 419 | 5.2          | 2.0          |
| concept    | 999 | 5.3          | 2.0          |
| identifier | 939 | 5.1          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 438   | 438      | 100% |
| astral-sh__uv      | 359   | 359      | 100% |
| badlogic__pi-mono  | 460   | 460      | 100% |
| django__django     | 672   | 672      | 100% |
| rbtr__rbtr         | 428   | 428      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 525   | 525      | 100% |
| concept    | 993   | 993      | 100% |
| docstring  | 314   | 314      | 100% |
| name       | 525   | 525      | 100% |

## Examples

### concept: `enableAllGitHubCopilotModels` (`badlogic__pi-mono`)

````typescript
Activate all available AI coding assistant models after user login to ensure they are ready for use
````

- **keywords:** activate_models, enable_assistants, model_init, assistant_startup,
  warmup_models
- **variants:** initialize all assistant models upon user authentication, trigger model
  loading and readiness check after login completes

### concept: `CustomProvidersStore` (`badlogic__pi-mono`)

````markdown
How to add and retrieve custom LLM providers like Ollama at runtime
````

- **keywords:** llm_provider, add_provider, get_provider, ollama, register_llm,
  custom_provider, provider_registry
- **variants:** dynamically register and fetch LLM backends, how to plug in a new model
  provider and look it up later

### concept: `CssPlugin` (`rbtr__rbtr`)

````python
Register stylesheet language support for parsing rules and tracking import dependencies
````

- **keywords:** register_language, stylesheet_parser, import_dependencies,
  language_support, parse_stylesheet
- **variants:** add a language processor for stylesheet files to resolve imports, enable
  a new stylesheet language with dependency tracking on file parse

### concept: `matcherFromTokens` (`django__django`)

````javascript
build an element matcher function from parsed selector tokens for CSS selector matching
````

- **keywords:** matches_selector, element_matches, selector_parser, compile_selector,
  match_element_by_token
- **variants:** function to check if a DOM element satisfies a parsed CSS selector AST,
  create a predicate from parsed CSS selector components to test node compliance

### concept: `Sivert Olstad, 2021` (`django__django`)

````markdown
Norwegian Nynorsk translation file for admin documentation template tags filters and views
````

- **keywords:** nn, nynorsk, django.po, admin_docs, translation
- **variants:** Nynorsk locale strings for admin templating and documentation views,
  Norwegian Nynorsk gettext catalog for admin docs filters and template tags

### identifier: `g` (`rbtr__rbtr`)

````typescript
// handle_status on the daemon calls git.open_repo, so the test
// repo has to be a real git directory with at least one commit.
````

- **keywords:** handle_status, daemon_status_handler, process_status,
  status_request_handler, handle_repo_status
- **variants:** daemon handler that opens a git repository and processes a status
  request, server endpoint that inspects the working tree status of a git repository

### identifier: `RenderedToolHtml` (`badlogic__pi-mono`)

````typescript
/** Pre-rendered HTML for a custom tool call and result */
````

- **keywords:** precomputed_html, cached_html_output, static_html_result,
  precomputed_html_result, template_html_render
- **variants:** pre-rendered HTML output for a custom tool call, statically generated
  HTML from a tool invocation

### identifier: `EmbedCompleteNotification` (`rbtr__rbtr`)

````typescript
EmbedCompleteNotification
````

- **keywords:** embed_completion_notification, EmbedFinishedEvent, EmbedReadyCallback,
  EmbedDoneAlert, EmbedRenderComplete
- **variants:** notify when embed rendering or processing finishes, signal that an
  embedded content has completed loading

### identifier: `frame_to_chunks` (`rbtr__rbtr`)

````python
frame_to_chunks
````

- **keywords:** split_frame, frame_to_blocks, decompose_frame, chunk_frame,
  frame_segmentation
- **variants:** split a data frame into smaller chunks for batch processing, break a
  frame into fixed-size segments or tiles

### identifier: `.toggle input:checked + .slider::before` (`anthropics__skills`)

````css
.toggle input:checked + .slider::before
````

- **keywords:** toggle_switch_checked_slider, checkbox_slider_before,
  switch_thumb_checked, toggle_thumb_pseudo, css_slider_thumb
- **variants:** style the thumb knob of a toggle switch when the checkbox is checked,
  CSS pseudo-element for the sliding knob of an on-off switch

### code: `6a27f10aef159701c7a5ff07f0fb0a78_05545ed <bc5d401a7ecd9343dd5afac265ed8ab3_4845>, 2011-2012,2014` (`django__django`)

````markdown
# 6a27f10aef159701c7a5ff07f0fb0a78_05545ed <bc5d401a7ecd9343dd5afac265ed8ab3_4845>, 2011-2012,2014
````

- **keywords:** copyright_year, 2012, 2014, license_header, date_range
- **variants:** source file copyright or license header spanning multiple years
  2011-2014, comment indicating copyright years including 2012 and 2014

### code: `showDialog` (`badlogic__pi-mono`)

````typescript
function showDialog(dialog: Component): void {
		activeDialog = dialog;
		setBottomComponent(dialog);
	}
````

- **keywords:** showDialog, activeDialog, setBottomComponent, dialog, Component
- **variants:** display a dialog by storing it as active and setting it as the bottom
  component, set current dialog and assign it to bottom panel, open dialog component and
  register as active bottom widget

### code: `extractAnnotations` (`anthropics__skills`)

````javascript
async function extractAnnotations() {
    const loadingTask = pdfjsLib.getDocument('annotated.pdf');
    const pdf = await loadingTask.promise;
````

- **keywords:** extractAnnotations, pdfjsLib, getDocument, loadingTask, pdfjs,
  annotations
- **variants:** load a PDF file and wait for the document promise using pdf.js, async
  function to open annotated.pdf with pdfjs and retrieve annotations

### code: `ctx.newSession(options?)` (`badlogic__pi-mono`)

````markdown
### ctx.newSession(options?)
````

- **keywords:** newSession, ctx, session, options, context
- **variants:** call newSession on the context with an optional options parameter,
  create a new context session with optional configuration

### code: `Id` (`astral-sh__uv`)

````rust
impl Deref for Id {
    type Target = str;
````

- **keywords:** Deref, Id, Target, str, deref_impl
- **variants:** implement Deref trait for Id wrapper targeting str, Id type dereferences
  to string slice, newtype Id with Deref to str

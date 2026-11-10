# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                                         |
| ------------- | --------------------------------------------- |
| model         | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| total queries | 3629                                          |
| expanded      | 3629 / 3629 (100%)                            |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 580  | 5.1          | 2.0          |
| concept    | 1614 | 5.1          | 2.0          |
| identifier | 1435 | 4.9          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 738   | 738      | 100% |
| astral-sh__uv      | 907   | 907      | 100% |
| badlogic__pi-mono  | 648   | 648      | 100% |
| django__django     | 747   | 747      | 100% |
| rbtr__rbtr         | 589   | 589      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 967   | 967      | 100% |
| concept    | 1564  | 1564     | 100% |
| docstring  | 389   | 389      | 100% |
| name       | 709   | 709      | 100% |

## Examples

### concept: `CustomBaseModel` (`django__django`)

````python
create a custom base model class for Django models
````

- **keywords:** AbstractModel, models.Model, TimeStampedModel, BaseModelMeta, class Meta
  abstract
- **variants:** define an abstract parent model that other models inherit from, write a
  mixin base model with common fields

### concept: `_normalize_arch` (`astral-sh__uv`)

````python
map an architecture name to its canonical form
````

- **keywords:** arch_name, canonicalize, normalize_arch, arch_alias, resolve_platform
- **variants:** normalize architecture string to standard name, map platform alias to
  official arch designation

### concept: `idx_chunks_repo_blob` (`rbtr__rbtr`)

````sql
how to index chunks by repository and blob hash
````

- **keywords:** chunk_id, blob_sha, repo_key, dedupe, content_hash_index
- **variants:** how are code fragments keyed by source file and git blob, what is the
  lookup key for indexing snippets per repository

### concept: `.inline-related h3 span.delete label` (`django__django`)

````css
align the delete label spacing in related items section
````

- **keywords:** margin, gap, flex_gap, text-align, padding, remove_icon, delete_button
- **variants:** make the remove button and its text line up consistently, adjust the
  delete control's horizontal alignment in the related records list

### concept: ```CheckboxSelectMultiple``` (`django__django`)

````rst
how to render a multi-select field as a list of checkboxes in Django forms
````

- **keywords:** CheckboxSelectMultiple, forms.CheckboxSelectMultiple,
  MultipleChoiceField, widget, render_checkbox_group
- **variants:** display many choices with individual checkbox widgets, generate multiple
  checkboxes for a field widget choice

### identifier: `NUMBER_GROUPING` (`django__django`)

````python
NUMBER_GROUPING
````

- **keywords:** number_format, format_number, group_digits, digit_grouping,
  locale_number, thousands_separator
- **variants:** group digits into thousands with separators, apply locale-aware
  thousands grouping to a numeric value

### identifier: `ROOT` (`astral-sh__uv`)

````python
ROOT = SELF_DIR.parent
````

- **keywords:** base_dir, PROJECT_ROOT, app_root, current_path_parent, BASE_PATH
- **variants:** get the parent directory of the current working directory, define the
  project base path from the script location

### identifier: `FakeReply` (`rbtr__rbtr`)

````typescript
FakeReply
````

- **keywords:** synthetic_response, mock_answer, dummy_reply, stub_response,
  placeholder_response
- **variants:** produce a pre-defined canned response instead of a real one, generate a
  non-functional simulated answer for testing

### identifier: `GeometryTypeControl` (`django__django`)

````javascript
GeometryTypeControl
````

- **keywords:** geometry_type_selector, GeometryKindControl, type_restriction_widget,
  geometry_class_filter, ShapeTypeControl
- **variants:** restrict which geometry kinds are allowed in a layer or input, widget to
  choose a category of geometric shapes

### identifier: `TIME_INPUT_FORMATS` (`django__django`)

````python
TIME_INPUT_FORMATS = [
    "%H:%M:%S",  # 14:30:59
    "%H:%M",  # 14:30
]
````

- **keywords:** TIME_FORMATS, time_format_config, parsed_time_formats,
  input_time_patterns, time_strptime_formats
- **variants:** list of strptime patterns used to parse input time strings, supported
  time parsing format patterns

### code: `Command` (`django__django`)

````python
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--list", action="store_true", help="Print all options")
````

- **keywords:** add_arguments, parser, BaseCommand, store_true, argparse
- **variants:** register CLI arguments onto the parser, set up a boolean flag option on
  the argument parser

### code: `manylinux_compatible` (`astral-sh__uv`)

````python
def manylinux_compatible(*_, **__):  # PEP 600
    return False
````

- **keywords:** manylinux_compatible, PEP 600, compatibility_check,
  unsupported_platform, policy
- **variants:** return that the platform is not manylinux compatible, stub policy
  function rejecting manylinux support

### code: `DaemonStatus` (`rbtr__rbtr`)

````typescript
type DaemonStatus = DaemonStatusReport;
````

- **keywords:** DaemonStatus, DaemonStatusReport, type_alias, type_definition
- **variants:** alias for the daemon status report type, declare DaemonStatus as an
  alias of DaemonStatusReport

### code: `--link-selected-fg` (`django__django`)

````css
--link-selected-fg: #6f94c6;
````

- **keywords:** link_selected_fg, selected_link_color, CSS_variable, foreground,
  custom_property
- **variants:** define the foreground color for selected links via a CSS custom
  property, set the selected link text color to #6f94c6

### code: `makefile` (`django__django`)

````python
def makefile(mode, *a, **kw):
            if mode == "rb":
                return rfile
            elif mode == "wb":
                return wfile
````

- **keywords:** makefile, rfile, wfile, open_mode, rb, wb
- **variants:** factory returning read or write file handle based on mode, dispatch to
  rfile for rb and wfile for wb modes

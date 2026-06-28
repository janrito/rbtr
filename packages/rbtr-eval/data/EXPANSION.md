# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                    |
| ------------- | ------------------------ |
| model         | `openai:zai-org/GLM-5.1` |
| total queries | 2684                     |
| expanded      | 2684 / 2684 (100%)       |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 414  | 5.1          | 2.0          |
| concept    | 1181 | 5.1          | 2.0          |
| identifier | 1089 | 5.1          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 224   | 224      | 100% |
| astral-sh__uv      | 628   | 628      | 100% |
| badlogic__pi-mono  | 504   | 504      | 100% |
| django__django     | 809   | 809      | 100% |
| rbtr__rbtr         | 519   | 519      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 650   | 650      | 100% |
| concept    | 1157  | 1157     | 100% |
| docstring  | 227   | 227      | 100% |
| name       | 650   | 650      | 100% |

## Examples

### concept: `ready` (`django__django`)

````javascript
run a callback when the DOM is finished loading
````

- **keywords:** DOMContentLoaded, onload, ready, addEventListener, document.readyState
- **variants:** execute a function after the page is ready, how to wait for DOM ready
  before running code

### concept: `tests/admin_views/templates/admin/admin_views/article/change_form_object_tools.html` (`django__django`)

````html
how to add an export button to Django change form object tools
````

- **keywords:** object_tools, change_form_object_tools, admin_change_form,
  export_action, ExtraButtonsMixin
- **variants:** customize Django admin change form toolbar with an export action, insert
  export link into Django admin object tools block

### concept: ```DIRS``` (`django__django`)

````rst
how to configure template search directories in Django
````

- **keywords:** TEMPLATES, DIRS, APP_DIRS, template_dirs, FileSystemLoader
- **variants:** where does Django look for template files, customize paths where Django
  searches for HTML templates

### concept: `area` (`django__django`)

````json
how to fix JSON parse error expected value at line 1 column 1
````

- **keywords:** JSON.parse, json.loads, JSONDecodeError, SyntaxError unexpected token,
  invalid_json
- **variants:** empty response body causes json parse failure at position 0, handling
  malformed json when decoding returns unexpected character at start

### concept: `test_readonly_stacked_inline_label` (`django__django`)

````python
Verify that readonly stacked inlines display the correct label in the admin change view
````

- **keywords:** StackedInline, readonly, InlineModelAdmin, change_view, label_for
- **variants:** Django admin stacked inline label rendering for read-only fields,
  correct label text shown on readonly inline in admin change form

### identifier: `len` (`astral-sh__uv`)

````rust
/// Return the number of `--find-links` entries.
````

- **keywords:** find_links_count, num_find_links, count_find_links, n_find_links,
  find_links_len
- **variants:** count the number of find-links sources, return count of --find-links
  option entries

### identifier: `renderCopyLinkButton` (`badlogic__pi-mono`)

````javascript
/**
       * Render the copy-link button HTML for a message.
````

- **keywords:** renderCopyLinkButton, copy_link_btn, message_link_button,
  create_share_link_markup, CopyLinkButton
- **variants:** render a button to copy a message's shareable URL, generate HTML for
  copying a permalink to a chat message

### identifier: `warn_file_conflict` (`astral-sh__uv`)

````rust
/// Check if all files are the same size, if so assume they are identical
````

- **keywords:** check_identical_by_size, files_same_size, are_files_identical,
  compare_file_sizes, size_based_equality
- **variants:** determine if files are identical by comparing their file sizes, verify
  all files are duplicates by checking matching sizes

### identifier: `the performance. Learn more at` (`rbtr__rbtr`)

````markdown
# the performance
````

- **keywords:** perf_metrics, performance_measurement, benchmark_results, perf_score,
  performance_eval
- **variants:** measure execution speed and resource usage, track and report system
  performance metrics

### identifier: `Consistency` (`django__django`)

````rst
Consistency
-----------
````

- **keywords:** consistency_model, data_consistency, consistency_check,
  EventualConsistency, consistency_validator
- **variants:** ensure data remains uniform across distributed systems, validate that
  replicated state matches across nodes

### code: `audit` (`astral-sh__uv`)

````json
"audit": {
      "anyOf": [
        {
          "$ref": "#/definitions/AuditOptions"
        },
        {
          "type": "null"
        }
      ]
    }
````

- **keywords:** audit, AuditOptions, anyOf, nullable_schema, definitions_ref
- **variants:** JSON Schema anyOf allowing AuditOptions or null, nullable audit property
  referencing AuditOptions definition

### code: `Thanh Le Viet <lethanhx2k@gmail.com>, 2013` (`django__django`)

````markdown
Thanh Le Viet <lethanhx2k@gmail.com>, 2013
````

- **keywords:** lethanhx2k, Thanh Le Viet, author_attribution, copyright_header, 2013
- **variants:** author comment header with email and year, source file copyright
  attribution by Thanh Le Viet

### code: `Filip Cuk <filipcuk2@gmail.com>, 2016` (`django__django`)

````markdown
# Filip Cuk <filipcuk2@gmail.com>, 2016
````

- **keywords:** Filip Cuk, filipcuk2, author_comment, copyright_header,
  translator_credits
- **variants:** author attribution comment with name and email year, translation file
  translator credit header

### code: `EmailBackend` (`django__django`)

````python
class EmailBackend(BaseEmailBackend):
    def send_messages(self, email_messages):
        return len(list(email_messages))
````

- **keywords:** EmailBackend, BaseEmailBackend, send_messages, email_messages,
  Django_email_backend
- **variants:** custom Django email backend that counts sent messages, def send_messages
  returning number of email messages sent

### code: `.ansi-line` (`badlogic__pi-mono`)

````css
.ansi-line {
      white-space: pre-wrap;
    }
````

- **keywords:** ansi-line, pre-wrap, white-space, terminal_output_css, ansi_formatting
- **variants:** CSS class for rendering ANSI terminal output with preserved whitespace
  and wrapping, style rule preserving whitespace formatting for ansi text lines

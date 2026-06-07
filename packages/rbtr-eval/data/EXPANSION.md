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

### concept: `CompressionMethod` (`astral-sh__uv`)

````rust
supported archive compression formats
````

- **keywords:** supported_formats, compression_codec, archive_type,
  compression_algorithm, file_ext
- **variants:** which compression codecs are available for archives, what archive types
  can be decompressed

### concept: `list_indexed_commits` (`rbtr__rbtr`)

````python
retrieve the commits that have been processed for a given repository with their timestamps
````

- **keywords:** processed_commits, commit_history, get_commits, commit_timestamps,
  fetch_revisions
- **variants:** list already-handled commits and their dates for a repo, get the log of
  ingested commits with time info for a repository

### concept: `search_p95_ms` (`rbtr__rbtr`)

````json
how to configure p95 search latency threshold
````

- **keywords:** p95_threshold, latency_threshold, percentile_latency, search_sla,
  response_time_config
- **variants:** set the 95th percentile response time limit for search, configure search
  performance SLA and latency bounds

### concept: `All models` (`anthropics__skills`)

````markdown
Model Migration Guide > Breaking Changes by Source Model.All models
````

- **keywords:** breaking_changes, migration_guide, deprecation, model_upgrade,
  source_model
- **variants:** what changed when switching between model versions, deprecation and
  migration notes per source model

### concept: `django/contrib/admindocs/templates/admin_doc/bookmarklets.html` (`django__django`)

````html
how to install browser shortcuts that navigate to the documentation for the current page's view
````

- **keywords:** bookmarklet, docs_bookmarklet, admindocs, install_shortcut,
  documentation_shortcut
- **variants:** set up bookmarklet that opens docs for the current view, add browser
  bookmarklet to jump to page documentation

### identifier: `Parameter schemas` (`rbtr__rbtr`)

````markdown
### Parameter schemas
````

- **keywords:** param_schema, param_spec, arg_schema, parameter_validation, param_defs
- **variants:** define validation rules and types for function or API parameters,
  specify the structure and constraints of input parameters

### identifier: `.tree-node` (`badlogic__pi-mono`)

````css
.tree-node
````

- **keywords:** treeNode, tree_item, hierarchy_node, tree_node_item, branch_node
- **variants:** a single element within a hierarchical tree view component, rendering a
  node in a collapsible tree data structure

### identifier: `Installation` (`badlogic__pi-mono`)

````markdown
@mariozechner/pi-agent-core.Installation
````

- **keywords:** install, setup, deployment, agent_instance, pi_agent_install
- **variants:** represent an installed pi-agent instance, manage pi-agent setup and
  deployment configuration

### identifier: `Embedding summary` (`rbtr__rbtr`)

````markdown
Embedding summary
````

- **keywords:** embedding_aggregation, embedding_pooling, summarize_embeddings,
  embedding_reduce, embedding_compression
- **variants:** aggregate multiple embedding vectors into a single summary vector,
  reduce dimensionality of embeddings into a compact representation

### identifier: `from_url` (`astral-sh__uv`)

````rust
/// Parse [`Credentials`] from a URL, if any.
````

- **keywords:** credentials_from_url, extract_credentials, parse_url_credentials,
  credential_from_uri, url_credentials
- **variants:** extract username and password from a URL, parse authentication
  credentials embedded in a URL

### code: `navigateTo` (`badlogic__pi-mono`)

````javascript
function navigateTo(targetId, scrollMode = 'target', scrollToEntryId = null) {
        currentLeafId = targetId;
        currentTargetId = scrollToEntryId || targetId;
        const path = getPath(tar
````

- **keywords:** navigateTo, targetId, scrollMode, scrollToEntryId, currentLeafId,
  getPath
- **variants:** function that navigates to a target node and scrolls to an entry within
  it, navigate to tree leaf with optional scroll-to-entry behavior

### code: `uv` (`astral-sh__uv`)

````toml
[tool.uv]
dev-dependencies = [
  "Sphinx",
  "asyncudp>=0.7",
  "black==24.4.2",
  "cairosvg",
  "celery-types",
  "coverage",
  "curlylint",
  "doc8>=1.1.0",
  "factory_boy",
  "flake8",
  "freezegun
````

- **keywords:** dev-dependencies, pyproject.toml, tool.uv, development_dependencies,
  Sphinx
- **variants:** uv dev-dependencies configuration in pyproject.toml, Python project
  development dependency list

### code: `_flavor_priority` (`astral-sh__uv`)

````python
def _flavor_priority(self, flavor: str) -> int:
        try:
            priority = self.FLAVOR_PREFERENCES.index(flavor)
        except ValueError:
            priority = len(self.FLAVOR_PREFERENCES)
````

- **keywords:** _flavor_priority, FLAVOR_PREFERENCES, priority, index, ValueError
- **variants:** get priority ranking for a flavor with fallback for unknown values,
  lookup flavor preference order defaulting to lowest priority

### code: `django/forms/templates/django/forms/formsets/ul.html` (`django__django`)

````html
{{ formset.management_form }}{% for form in formset %}{{ form.as_ul }}{% endfor %}
````

- **keywords:** formset, management_form, as_ul, django_formset, management_form
- **variants:** render a Django formset with management form and each form as unordered
  list, Django template loop over formset forms rendering management_form and form.as_ul

### code: ```WeekArchiveView``` (`django__django`)

````rst
``WeekArchiveView``
===================
````

- **keywords:** WeekArchiveView, archive_week, DateMixin, WeekArchiveMixin, django_views
- **variants:** Django generic view for weekly archive listing, class-based view that
  shows objects by week

# Expansion report

LLM-generated keyword synonyms and variant rephrasings for
search queries. Every query kind receives both keywords and
variants; the prompt is tailored per kind. The downstream
ablation in `measure` isolates the effect of each channel.

## Summary

| field         | value                         |
| ------------- | ----------------------------- |
| model         | `openai-chat:zai-org/GLM-5.2` |
| total queries | 3618                          |
| expanded      | 3618 / 3618 (100%)            |

## Per-kind breakdown

| query_kind | n    | avg_keywords | avg_variants |
| ---------- | ---- | ------------ | ------------ |
| code       | 540  | 5.3          | 2.0          |
| concept    | 1565 | 5.4          | 2.0          |
| identifier | 1513 | 5.3          | 2.0          |

## Per-repo breakdown

| slug               | total | expanded | rate |
| ------------------ | ----- | -------- | ---- |
| anthropics__skills | 734   | 734      | 100% |
| astral-sh__uv      | 895   | 895      | 100% |
| badlogic__pi-mono  | 660   | 660      | 100% |
| django__django     | 766   | 766      | 100% |
| rbtr__rbtr         | 563   | 563      | 100% |

## Per-provenance breakdown

| provenance | total | expanded | rate |
| ---------- | ----- | -------- | ---- |
| body       | 913   | 913      | 100% |
| concept    | 1523  | 1523     | 100% |
| docstring  | 389   | 389      | 100% |
| name       | 793   | 793      | 100% |

## Examples

### concept: `on_download_progress` (`astral-sh__uv`)

````rust
Callback handler for reporting download progress updates by bytes downloaded
````

- **keywords:** on_progress, download_callback, progress_handler, bytes_received,
  tqdm_callback
- **variants:** progress bar hook for tracking bytes downloaded during file fetch,
  monitor download byte count via callback notification

### concept: `docs/_theme/djangodocs/static/default.css` (`django__django`)

````css
how to style documentation pages with custom CSS
````

- **keywords:** custom_css, stylesheet, extra_css, theme_css, docs_styling
- **variants:** apply custom stylesheets to documentation site, override default
  documentation theme with CSS rules

### concept: `_make_api_request` (`anthropics__skills`)

````python
Send an async HTTP request to an API endpoint and return the JSON response
````

- **keywords:** aiohttp, fetch_json, httpx, async_request, await response.json()
- **variants:** make a non-blocking HTTP call and parse the response body as JSON, how
  to perform an asynchronous API fetch and deserialize the result

### concept: `.section` (`anthropics__skills`)

````css
Style a container block with background, border, and rounded corners using CSS custom properties
````

- **keywords:** border-radius, background-color, border, var(--, :root, custom-property,
  card styling
- **variants:** apply CSS variables to theme a styled card with rounded edges and
  background, use CSS custom properties for container theming with border and corner
  radius

### concept: `chat` (`anthropics__skills`)

````typescript
Send a message to Claude and get the text response back
````

- **keywords:** claude_completion, anthropic_api, send_message, claude_response,
  chat_completion
- **variants:** call Anthropic Claude API and retrieve the returned text, send a prompt
  to Claude and parse the text output from the response

### identifier: `MCPConnectionSSE` (`anthropics__skills`)

````python
"""MCP connection using Server-Sent Events."""
````

- **keywords:** sse_connection, McpSseClient, sse_transport, EventSourceClient,
  mcp_sse_session, SSEClient
- **variants:** establish MCP server connection over Server-Sent Events transport,
  connect to MCP server using SSE event stream

### identifier: `hexToRgb` (`anthropics__skills`)

````javascript
// Color utilities
````

- **keywords:** color_helpers, colorUtils, colour_helpers, palette_utils, color_tools
- **variants:** utility functions for color manipulation and conversion, parse, convert,
  and format colors between RGB HSV and hex

### identifier: `csrf` (`django__django`)

````python
csrf
````

- **keywords:** xsrf, cross_site_request_forgery, anti_csrf, csrf_token, CSRFMiddleware
- **variants:** generate and validate tokens to prevent cross-site request forgery
  attacks, middleware that blocks forged POST requests from unauthorized origins

### identifier: `CLAUDE.md:1-2` (`astral-sh__uv`)

````markdown
CLAUDE.md:1-2
````

- **keywords:** project_readme, assistant_instructions, config_guidelines, context_file,
  project_manifest
- **variants:** project-level configuration file with instructions for an AI assistant,
  documentation file outlining coding conventions and project context for Claude

### identifier: `submit_row` (`django__django`)

````python
submit_row
````

- **keywords:** submitRecord, submit_entry, insertRow, saveRow, commit_row, addRow
- **variants:** submit a single table row for processing or persistence, send one data
  row to a backend service for storage

### code: `imageData` (`anthropics__skills`)

````typescript
imageData = fs.readFileSync("image.png").toString("base64")
````

- **keywords:** readFileSync, base64, imageData, toString, encode_image_base64
- **variants:** read an image file and encode its contents as a base64 string, load
  image.png from disk and convert to base64 representation

### code: `response` (`anthropics__skills`)

````python
# Default to Opus for most tasks
response = client.messages.create(
    model="claude-opus-4-7",  # $5.00/$25.00 per 1M tokens
    max_tokens=16000,
    messages=[{"role": "user", "content": "Explain
````

- **keywords:** client.messages.create, claude-opus-4-7, max_tokens, messages, Anthropic
  API
- **variants:** Send a chat request to Claude Opus using the Anthropic Messages API with
  a 16000 token limit, Create a completion via the Anthropic Python SDK specifying the
  claude-opus-4-7 model and max_tokens

### code: `mayCreate` (`badlogic__pi-mono`)

````javascript
mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")}
````

- **keywords:** mayCreate, FS.lookupNode, FS.isDir, FS.nodePermissions, EEXIST, EACCES
- **variants:** check permission to create a filesystem node in a directory returning
  error code if it exists or lacks write/execute permission, function mayCreate(dir,
  name) returns 54 if not directory, 20 if name already exists, otherwise checks wx
  permissions

### code: `test_extended_bodyclass_template_login` (`django__django`)

````python
def test_extended_bodyclass_template_login(self):
        """
        The admin/login.html template uses block.super in the
        bodyclass block
````

- **keywords:** test_extended_bodyclass_template_login, admin/login.html, block.super,
  bodyclass, template_login
- **variants:** test that admin login template extends bodyclass block using
  block.super, verify admin login HTML template inherits and augments the bodyclass
  block

### code: `update` (`anthropics__skills`)

````javascript
update() {
        // Update entity state
        // This might involve:
        // - Physics calculations
        // - Behavioral rules
        // - Interactions with neighbors
    }
````

- **keywords:** update, entity_state, physics, behavioral_rules, neighbor_interactions
- **variants:** per-tick entity update applying physics, behavioral rules, and neighbor
  interactions, advance entity state each frame with movement, AI rules, and nearby
  entity checks

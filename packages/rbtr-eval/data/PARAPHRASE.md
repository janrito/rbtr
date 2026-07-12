# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                         |
| --------------- | ----------------------------- |
| model           | `openai-chat:zai-org/GLM-5.2` |
| concept queries | 993                           |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 174 |
| astral-sh__uv      | 159 |
| badlogic__pi-mono  | 198 |
| django__django     | 297 |
| rbtr__rbtr         | 165 |

## Per language

| language   | n   |
| ---------- | --- |
| markdown   | 240 |
| python     | 204 |
| javascript | 145 |
| css        | 131 |
| typescript | 99  |
| rust       | 63  |
| rst        | 48  |
| html       | 34  |
| sql        | 29  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `find_runs` (`anthropics__skills`)

````python
def find_runs(workspace: Path) -> list[dict]:
    """Recursively find directories that contain an outputs/ subdirectory."""
    runs: list[dict] = []
    _find_runs_recursive(workspace, workspace, runs)
    runs.sort(key=lambda r: (r.get("eval_id", float("inf")), r["id"]))
    return runs
````

> **concept:** find all directories containing an outputs subdirectory in a workspace

### `Build backend for uv` (`astral-sh__uv`)

````markdown
# Build backend for uv

This package is a slimmed down version of uv containing only the build backend. See
https://pypi.org/project/uv/ and https://docs.astral.sh/uv/ for the main project package and
documentation.
````

> **concept:** uv build backend package for Python projects

### `parse_changelog` (`astral-sh__uv`)

````python
def parse_changelog(content):
    """Parse the changelog content into individual version blocks."""
    # Use regex to split the content by version headers
    version_pattern = r"(?=## \d+\.\d+\.\d+)"
    version_blocks = re.split(version_pattern, content)

    # First item in the list is the header, which we want to preserve
    header = version_blocks[0]
    version_blocks = version_blocks[1:]

    return header, version_blocks
````

> **concept:** Split changelog text into individual version sections using regex

### `.aligned .form-row > div` (`django__django`)

````css
.aligned .form-row > div {
        width: calc(100vw - 30px);
    }
````

> **concept:** Make form row child elements fill the viewport width minus padding

### `findPosX` (`django__django`)

````javascript
// ----------------------------------------------------------------------------
// Find-position functions by PPK
// See https://www.quirksmode.org/js/findpos.html
// ----------------------------------------------------------------------------
function findPosX(obj) {
    let curleft = 0;
    if (obj.offsetParent) {
        while (obj.offsetParent) {
            curleft += obj.offsetLeft - obj.scrollLeft;
            obj = obj.offsetParent;
        }
    } else if (obj.x) {
        curleft += obj.x;
    }
    return curleft;
}
````

> **concept:** Calculate the horizontal pixel position of an element relative to the
> page

### `Jure Cuhalev <gandalf@owca.info>, 2012` (`django__django`)

````markdown
# Jure Cuhalev <gandalf@owca.info>, 2012
````

> **concept:** how to find the maintainer information and contact for this project

### `Consistency` (`django__django`)

````rst
Consistency
-----------

The framework should be consistent at all levels. Consistency applies to
everything from low-level (the Python coding style used) to high-level (the
"experience" of using Django).
````

> **concept:** principles for keeping a framework consistent across coding conventions
> and developer experience

### `test_password_changed_with_custom_validator` (`django__django`)

````python
def test_password_changed_with_custom_validator(self):
        class Validator:
            def password_changed(self, password, user):
                self.password = password
                self.user = user

        user = object()
        validator = Validator()
        password_changed("password", user=user, password_validators=(validator,))
        self.assertIs(validator.user, user)
        self.assertEqual(validator.password, "password")
````

> **concept:** verify custom password validator is called with correct user and password
> when password is changed

### `send` (`rbtr__rbtr`)

````typescript
/**
   * Send a request through the cached RPC endpoint.
   *
   * Re-queries once on transport failure (covers daemon restart
   * with a new endpoint) before giving up with
   * ``DaemonUnavailableError``.  ``RbtrDaemonError`` from the
   * daemon itself is propagated as-is — the caller decides what
   * to do with it.
   */
  async send<R extends Request>(request: R): Promise<ResponseFor<R["kind"]>> {
    if (!this.rpcEndpoint) {
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError("no rpc endpoint");
      }
    }

    try {
      return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
    } catch (err) {
      if (err instanceof Error && err.name === "RbtrDaemonError") {
        throw err;
      }
      // Transport failure — try one refresh + retry before
      // giving up.  Covers daemon restart with a new endpoint.
      this.status = null;
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError(err);
      }
      try {
        return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
      } catch (err2) {
        if (err2 instanceof Error && err2.name === "RbtrDaemonError") {
          throw err2;
        }
        throw new DaemonUnavailableError(err2);
      }
    }
  }
````

> **concept:** Dispatch an RPC request to the daemon with one automatic retry on
> transport failure before giving up

### `make_content_response` (`rbtr__rbtr`)

````python
def make_content_response(content: str = "") -> CreateChatCompletionResponse:
    """Build a plain chat-completion response with text content only."""
    return {
        "id": "stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
````

> **concept:** Create a mock chat completion response with simple text content

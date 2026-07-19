# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                         |
| --------------- | ----------------------------- |
| model           | `openai-chat:zai-org/GLM-5.2` |
| concept queries | 1523                          |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 300 |
| astral-sh__uv      | 385 |
| badlogic__pi-mono  | 275 |
| django__django     | 338 |
| rbtr__rbtr         | 225 |

## Per language

| language   | n   |
| ---------- | --- |
| python     | 367 |
| typescript | 205 |
| javascript | 171 |
| css        | 148 |
| markdown   | 127 |
| rust       | 118 |
| json       | 98  |
| bash       | 92  |
|            | 47  |
| yaml       | 39  |
| toml       | 34  |
| sql        | 31  |
| html       | 26  |
| rst        | 20  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `CreateUserInput` (`anthropics__skills`)

````python
class CreateUserInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    name: str = Field(..., description="User's full name", min_length=1, max_length=100)
    email: str = Field(..., description="User's email address", pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., description="User's age", ge=0, le=150)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Email cannot be empty")
        return v.lower()
````

> **concept:** validate user registration data with name, email, and age constraints

### `cfg(all(target_env = "msvc", target_os = "windows"))` (`astral-sh__uv`)

````toml
# statically link the C runtime so the executable does not depend on
# that shared/dynamic library.
#
# See: https://github.com/astral-sh/ruff/issues/11503
[target.'cfg(all(target_env = "msvc", target_os = "windows"))']
rustflags = ["-C", "target-feature=+crt-static"]

````

> **concept:** how to statically link the C runtime on Windows MSVC builds

### `<anonymous>` (`astral-sh__uv`)

````rust
// disable all rust entry points, requires enabling compiler-builtins-mem
````

> **concept:** How to disable Rust entry points and enable compiler-builtins-mem feature

### `restoreEditor` (`badlogic__pi-mono`)

````typescript
// Restore editor helper
		const restoreEditor = () => {
			this.editorContainer.clear();
			this.editorContainer.addChild(this.editor);
			this.ui.setFocus(this.editor);
			this.ui.requestRender();
		};
````

> **concept:** Restore the editor component back into its container and set focus

### `[0.63.1] - 2026-03-27` (`badlogic__pi-mono`)

````markdown
## [0.63.1] - 2026-03-27
````

> **concept:** What changed in the latest patch release updates

### `<anonymous>` (`django__django`)

````javascript
// Call the preDispatch hook for the mapped type, and let it bail if desired
````

> **concept:** how to call preDispatch hook before event dispatch

### `ImportedModelBackend` (`django__django`)

````python
class ImportedModelBackend(ModelBackend):
    pass
````

> **concept:** custom authentication backend that extends Django's default model backend

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

> **concept:** Generate a simulated Python object id to test sender memory address reuse
> with non-overlapping lifetimes

### `runtime_dir` (`rbtr__rbtr`)

````python
def runtime_dir(self) -> Path:
        """Per-`data_dir` runtime dir for sockets + status file.

        Keyed on `hash(resolve(data_dir))` so two daemons
        against different data dirs get independent runtime
        dirs.  Lives under `platformdirs.user_runtime_path('rbtr')`.
        """
        base = platformdirs.user_runtime_path(RBTR_NAME, ensure_exists=True)
        key = hashlib.sha256(str(self.data_dir.resolve()).encode()).hexdigest()[:16]
        path = base / key
        path.mkdir(parents=True, exist_ok=True)
        return path
````

> **concept:** Get a unique per-data-directory runtime path for daemon sockets and
> status files

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

> **concept:** create a fake chat completion API response with stub text content for
> testing

# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                                         |
| --------------- | --------------------------------------------- |
| model           | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| concept queries | 1558                                          |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 304 |
| astral-sh__uv      | 399 |
| badlogic__pi-mono  | 282 |
| django__django     | 331 |
| rbtr__rbtr         | 242 |

## Per language

| language   | n   |
| ---------- | --- |
| python     | 365 |
| typescript | 205 |
| javascript | 179 |
| bash       | 168 |
| css        | 145 |
| rust       | 119 |
| markdown   | 99  |
| json       | 97  |
| yaml       | 40  |
| rst        | 34  |
| sql        | 31  |
| plaintext  | 30  |
| toml       | 29  |
| html       | 17  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `c_signal` (`django__django`)

````python
c_signal = Signal()
````

> **concept:** how to create a new signal instance

### `Python` (`astral-sh__uv`)

````rust
/// The trampoline should just execute Python, it's a proxy Python executable.
    Python
````

> **concept:** what does the trampoline executable do in a proxy setup

### `chunks` (`rbtr__rbtr`)

````sql
-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:head_sha:'def'
-- sqlfluff:templater:placeholder:base_sha:'abc'
SELECT
  c.id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.language,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.repo_id = fs.repo_id
    AND c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.repo_id = $repo_id
  AND fs.commit_sha = $head_sha
  AND fs.file_path NOT IN (
    SELECT file_snapshots.file_path FROM file_snapshots
    WHERE
      file_snapshots.repo_id = $repo_id
      AND file_snapshots.commit_sha = $base_sha
  )
ORDER BY c.file_path, c.line_start
````

> **concept:** select code snippets from files that were added in the latest commit

### `o` (`django__django`)

````javascript
function o(e){if(C.documentMode){var t=_.get(this,"handle"),n=ce.event.fix(e);n.type="focusin"===e.type?"focus":"blur",n.isSimulated=!0,t(e),n.target===n.currentTarget&&t(n)}else ce.event.simulate(i,e.target,ce.event.fix(e))}
````

> **concept:** simulate focus and blur events for older browsers

### `DOWNLOAD_PREFIX` (`django__django`)

````bash
DOWNLOAD_PREFIX="https://www.djangoproject.com/download"
````

> **concept:** where to find the base URL for downloading Django releases

### `SessionTreeNode` (`badlogic__pi-mono`)

````typescript
/** A session tree node for hierarchical display */
interface SessionTreeNode {
	session: SessionInfo;
	children: SessionTreeNode[];
}
````

> **concept:** represent hierarchical session data in a tree structure

### `__init__` (`anthropics__skills`)

````python
def __init__(self, command: str, args: list[str] = None, env: dict[str, str] = None):
        super().__init__()
        self.command = command
        self.args = args or []
        self.env = env
````

> **concept:** Find code that initializes a server connection with a command line
> program

### `` (`badlogic__pi-mono`)

````javascript
// If a new directory is created, explicitly watch it
  // This ensures newly created artifact folders are monitored without restart
````

> **concept:** how to watch newly created directories for file monitoring

### `name` (`astral-sh__uv`)

````yaml
# Publish a release to crates.io.
#
# Assumed to run as a subworkflow of .github/workflows/release.yml; specifically, as a publish job
# within `cargo-dist`.
name: "Publish to crates.io"
````

> **concept:** publishing a crate release to crates.io as part of the release workflow

### `SearchRequest` (`rbtr__rbtr`)

````python
class SearchRequest(BaseModel):
    """Search the code index.

    `alpha` / `beta` / `gamma` override the per-`QueryKind`
    fusion weights for the duration of the call.  All-or-nothing:
    either all three are supplied (override applies uniformly
    across query kinds) or none are (per-kind defaults apply).
    When supplied they must each be in `[0.0, 1.0]` and sum to
    `1.0` within `1e-6`.
    """

    model_config = _STRICT
    kind: Literal["search"] = "search"
    path: str
    query: str
    limit: int = 10
    ref: str | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    gamma: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        supplied = [w for w in (self.alpha, self.beta, self.gamma) if w is not None]
        if not supplied:
            return self
        if len(supplied) != 3:
            msg = "alpha, beta, gamma must all be supplied together (or none)"
            raise ValueError(msg)
        total = supplied[0] + supplied[1] + supplied[2]
        if abs(total - 1.0) > 1e-6:
            msg = f"alpha + beta + gamma must sum to 1.0; got {total:.6f}"
            raise ValueError(msg)
        return self
````

> **concept:** model representing a request to search the code index with optional
> fusion weights

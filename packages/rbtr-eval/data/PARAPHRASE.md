# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                         |
| --------------- | ----------------------------- |
| model           | `openai-chat:zai-org/GLM-5.2` |
| concept queries | 1526                          |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 303 |
| astral-sh__uv      | 384 |
| badlogic__pi-mono  | 274 |
| django__django     | 341 |
| rbtr__rbtr         | 224 |

## Per language

| language   | n   |
| ---------- | --- |
| python     | 371 |
| typescript | 205 |
| javascript | 171 |
| css        | 148 |
| markdown   | 128 |
| rust       | 117 |
| json       | 96  |
| bash       | 93  |
|            | 50  |
| yaml       | 38  |
| toml       | 33  |
| sql        | 29  |
| html       | 27  |
| rst        | 20  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `handleApiError` (`anthropics__skills`)

````typescript
function handleApiError(error: unknown): string {
  if (error instanceof AxiosError) {
    if (error.response) {
      switch (error.response.status) {
        case 404:
          return "Error: Resource not found. Please check the ID is correct.";
        case 403:
          return "Error: Permission denied. You don't have access to this resource.";
        case 429:
          return "Error: Rate limit exceeded. Please wait before making more requests.";
        default:
          return `Error: API request failed with status ${error.response.status}`;
      }
    } else if (error.code === "ECONNABORTED") {
      return "Error: Request timed out. Please try again.";
    }
  }
  return `Error: Unexpected error occurred: ${error instanceof Error ? error.message : String(error)}`;
}
````

> **concept:** Convert HTTP and network errors into human-readable error messages

### `.github/renovate.json5:181-196` (`astral-sh__uv`)

````text
customType: "regex",
      managerFilePatterns: ["/(^|/)rust-toolchain\\.toml?$/"],
      matchStrings: [
        'channel\\s*=\\s*"(?<currentValue>\\d+\\.\\d+(\\.\\d+)?)"',
      ],
      depNameTemplate: "rust",
      packageNameTemplate: "rust-lang/rust",
      datasourceTemplate: "github-releases",
    },
  ],
  vulnerabilityAlerts: {
    commitMessageSuffix: "",
    labels: ["internal", "security"],
  },
}
````

> **concept:** how to configure custom regex pattern matching for rust toolchain version
> updates

### `with_cache_on_nocow_fs` (`astral-sh__uv`)

````rust
/// Use a cache directory on the filesystem specified by
    /// [`EnvVars::UV_INTERNAL__TEST_NOCOW_FS`].
    ///
    /// Returns `Ok(None)` if the environment variable is not set.
    pub fn with_cache_on_nocow_fs(self) -> anyhow::Result<Option<Self>> {
        let Some(dir) = env::var(EnvVars::UV_INTERNAL__TEST_NOCOW_FS).ok() else {
            return Ok(None);
        };
        self.with_cache_on_fs(&dir, "NOCOW_FS").map(Some)
    }
````

> **concept:** Configure cache to use a filesystem directory specified by an environment
> variable for no copy-on-write testing

### `SocketModeClient` (`badlogic__pi-mono`)

````javascript
{ SocketModeClient } = require('@slack/socket-mode')
````

> **concept:** how to import the Slack socket mode client using require

### `processOutput` (`badlogic__pi-mono`)

````typescript
// Process log output line by line
	const processOutput = (data: Buffer) => {
		const lines = data.toString().split("\n");
		for (const line of lines) {
			if (line) {
				console.log(line); // Echo the line to console

				// Check for startup complete message
				if (line.includes("Application startup complete")) {
					startupComplete = true;
					logProcess.kill(); // Stop tailing logs
				}

				// Check for failure indicators
				if (line.includes("Model runner exiting with code") && !line.includes("code 0")) {
					startupFailed = true;
					failureReason = "Model runner failed to start";
					logProcess.kill();
				}
				if (line.includes("Script exited with code") && !line.includes("code 0")) {
					startupFailed = true;
					failureReason = "Script failed to execute";
					logProcess.kill();
				}
				if (line.includes("torch.OutOfMemoryError") || line.includes("CUDA out of memory")) {
					startupFailed = true;
					failureReason = "Out of GPU memory (OOM)";
					// Don't kill immediately - let it show more error context
				}
				if (line.includes("RuntimeError: Engine core initialization failed")) {
					startupFailed = true;
					failureReason = "vLLM engine initialization failed";
					logProcess.kill();
				}
			}
		}
	};
````

> **concept:** Parse log output line by line to detect startup success or failure
> conditions like OOM errors and exit codes

### `--selected-row` (`django__django`)

````css
--selected-row: #00363a;
````

> **concept:** how to set custom theme variable for highlighted table row color

### `ImportedModelBackend` (`django__django`)

````python
class ImportedModelBackend(ModelBackend):
    pass
````

> **concept:** Django model authentication backend imported from external module

### `tests/fixtures_regress/fixtures/sequence_extra_yaml.yaml` (`django__django`)

````yaml

````

> **concept:** how to configure settings using YAML format

### `EmbedJob` (`rbtr__rbtr`)

````python
class EmbedJob(BaseModel):
    """An embed-index job for the unified work queue."""

    model_config = _STRICT
    kind: Literal["embed"] = "embed"
    path: str
    repo_id: int
    ref: str

    @property
    def dedupe_key(self) -> Hashable:
        return (self.repo_id, self.ref)
````

> **concept:** Queue job definition for embedding repository content at a specific ref

### `meta` (`rbtr__rbtr`)

````sql
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
)
````

> **concept:** create a simple key-value configuration table in the database

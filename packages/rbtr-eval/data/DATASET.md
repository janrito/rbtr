# rbtr search-quality dataset

Characterisation of the query set **before** measurement — the input
to the benchmark, not its results. 3618 queries across
5 repos and 14 languages.

The three axes are independent: `symbol_kind` is the target chunk,
`provenance` is how the query was generated, and `query_kind` is
`classify_query(text)` — the request shape search routes on.

## Repos

The indexed commit and sampled sizes per repo.

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 4155    | 434             |
| `astral-sh__uv`      | `cfe5277bc422` | 28197   | 510             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 23980   | 385             |
| `django__django`     | `e78a46a8fb29` | 67009   | 428             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 5922    | 338             |

## Queries per language

| language   | n   |
| ---------- | --- |
| python     | 898 |
| typescript | 521 |
| javascript | 431 |
| css        | 355 |
| markdown   | 281 |
| rust       | 248 |
| bash       | 233 |
| json       | 200 |
|            | 107 |
| yaml       | 92  |
| sql        | 80  |
| toml       | 74  |
| html       | 58  |
| rst        | 40  |

## Target coverage — `symbol_kind` × `provenance`

Which kinds of chunk the queries target, and how each splits across
generation strategies. A kind absent here is not measured.

| symbol_kind | body | concept | docstring | name | total |
| ----------- | ---- | ------- | --------- | ---- | ----- |
| variable    | 174  | 295     | 76        | 177  | 722   |
| class       | 136  | 251     | 110       | 137  | 634   |
| function    | 130  | 244     | 122       | 130  | 626   |
| method      | 100  | 164     | 60        | 100  | 424   |
| config_key  | 80   | 157     | 21        | 80   | 338   |
| comment     | 167  | 163     | 0         | 0    | 330   |
| raw_chunk   | 56   | 119     | 0         | 99   | 274   |
| doc_section | 70   | 130     | 0         | 70   | 270   |

## Target × request shape — `symbol_kind` × `query_kind`

For each target kind, the request shapes generated against it. Both
axes are independent of provenance.

| symbol_kind | concept | identifier | code | total |
| ----------- | ------- | ---------- | ---- | ----- |
| variable    | 313     | 296        | 113  | 722   |
| class       | 246     | 254        | 134  | 634   |
| function    | 243     | 253        | 130  | 626   |
| method      | 165     | 170        | 89   | 424   |
| config_key  | 158     | 157        | 23   | 338   |
| comment     | 178     | 143        | 9    | 330   |
| raw_chunk   | 122     | 137        | 15   | 274   |
| doc_section | 140     | 103        | 27   | 270   |

## Not measured

No queries are generated for these chunk kinds: `import`.

Languages skipped for having fewer measurable chunks than the
threshold:

| slug                 | language     | n_chunks |
| -------------------- | ------------ | -------- |
| `anthropics__skills` | `go`         | 44       |
| `anthropics__skills` | `html`       | 10       |
| `anthropics__skills` | `java`       | 48       |
| `anthropics__skills` | `rst`        | 27       |
| `anthropics__skills` | `ruby`       | 35       |
| `astral-sh__uv`      | `c`          | 3        |
| `astral-sh__uv`      | `javascript` | 41       |
| `astral-sh__uv`      | `rst`        | 5        |
| `badlogic__pi-mono`  | ``           | 10       |
| `badlogic__pi-mono`  | `c`          | 14       |
| `badlogic__pi-mono`  | `html`       | 6        |
| `badlogic__pi-mono`  | `python`     | 7        |
| `badlogic__pi-mono`  | `yaml`       | 32       |
| `django__django`     | `bash`       | 22       |
| `django__django`     | `toml`       | 9        |
| `rbtr__rbtr`         | ``           | 36       |
| `rbtr__rbtr`         | `bash`       | 28       |
| `rbtr__rbtr`         | `css`        | 2        |
| `rbtr__rbtr`         | `javascript` | 1        |
| `rbtr__rbtr`         | `rust`       | 1        |
| `rbtr__rbtr`         | `toml`       | 45       |
| `rbtr__rbtr`         | `yaml`       | 13       |

## Classification — `provenance` × `query_kind`

How each generation strategy's text classifies as a request shape
(row-normalised). Provenance and query_kind are different axes: the
scatter here is that difference made visible.

| provenance  | concept | identifier | code  | n    |
| ----------- | ------- | ---------- | ----- | ---- |
| `body`      | 3.5%    | 40.0%      | 56.5% | 913  |
| `concept`   | 98.6%   | 1.4%       | 0.0%  | 1523 |
| `docstring` | 5.9%    | 90.0%      | 4.1%  | 389  |
| `name`      | 1.0%    | 98.0%      | 1.0%  | 793  |

## Examples

Sampled queries per provenance — the actual text fed to search,
verbatim.

**`body` → code** · python · `class` · PyodideFinder

````text
class PyodideFinder(Finder):
    implementation = ImplementationName.CPYTHON
````

**`body` → code** · rust · `class` · RevisionId

````text
impl AsRef<Path> for RevisionId {
    fn as_ref(&self) -> &Path {
        self.0.as_ref()
    }
}
````

**`body` → identifier** · python · `comment` · <anonymous>

````text
# via
    #   -c requirements.txt
    #   pydantic
````

**`concept` → concept** · rust · `variable` · Python

````text
trampoline acting as a proxy executable to launch the interpreter
````

**`concept` → concept** · css · `variable` · --proton

````text
how to define a custom CSS color variable with a hex value
````

**`concept` → concept** · toml · `raw_chunk` · rustfmt.toml

````text
how to configure formatting settings in a TOML configuration file
````

**`docstring` → identifier** · python · `variable` · USE_UV_EXECUTABLE

````text
# Use the `uv build-backend` command rather than `uv-build`.
````

**`docstring` → identifier** · rust · `function` · extra_build_requires_for

````text
/// Determine the extra build requirements for the given package name.
````

**`docstring` → identifier** · rust · `class` · PrioritizedDist

````text
/// Create a new [`PrioritizedDist`] from the given wheel distribution.
````

**`name` → identifier** · python · `method` · key

````text
PythonDownload::key
````

**`name` → identifier** · rust · `class` · Username

````text
Username
````

**`name` → identifier** · rust · `method` · fmt

````text
MarkerValue::fmt
````

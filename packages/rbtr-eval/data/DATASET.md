# rbtr search-quality dataset

Characterisation of the query set **before** measurement — the input
to the benchmark, not its results. 3623 queries across
5 repos and 14 languages.

The three axes are independent: `symbol_kind` is the target chunk,
`provenance` is how the query was generated, and `query_kind` is
`classify_query(text)` — the request shape search routes on.

## Repos

The indexed commit and sampled sizes per repo.

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 5462    | 431             |
| `astral-sh__uv`      | `cfe5277bc422` | 27256   | 505             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 24302   | 369             |
| `django__django`     | `e78a46a8fb29` | 67406   | 415             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 5973    | 345             |

## Queries per language

| language   | n   |
| ---------- | --- |
| python     | 885 |
| typescript | 516 |
| javascript | 431 |
| bash       | 389 |
| css        | 349 |
| rust       | 249 |
| markdown   | 199 |
| json       | 197 |
| yaml       | 92  |
| sql        | 80  |
| rst        | 69  |
| toml       | 68  |
| plaintext  | 60  |
| html       | 39  |

## Target coverage — `symbol_kind` × `provenance`

Which kinds of chunk the queries target, and how each splits across
generation strategies. A kind absent here is not measured.

| symbol_kind | body | concept | docstring | name | total |
| ----------- | ---- | ------- | --------- | ---- | ----- |
| variable    | 182  | 307     | 76        | 186  | 751   |
| function    | 131  | 257     | 122       | 131  | 641   |
| class       | 136  | 257     | 110       | 137  | 640   |
| method      | 100  | 175     | 60        | 100  | 435   |
| doc_section | 130  | 198     | 0         | 75   | 403   |
| comment     | 178  | 178     | 0         | 0    | 356   |
| config_key  | 80   | 156     | 21        | 80   | 337   |
| raw_chunk   | 30   | 30      | 0         | 0    | 60    |

## Target × request shape — `symbol_kind` × `query_kind`

For each target kind, the request shapes generated against it. Both
axes are independent of provenance.

| symbol_kind | concept | identifier | code | total |
| ----------- | ------- | ---------- | ---- | ----- |
| variable    | 322     | 313        | 116  | 751   |
| function    | 257     | 252        | 132  | 641   |
| class       | 257     | 247        | 136  | 640   |
| method      | 175     | 171        | 89   | 435   |
| doc_section | 205     | 146        | 52   | 403   |
| comment     | 190     | 154        | 12   | 356   |
| config_key  | 157     | 152        | 28   | 337   |
| raw_chunk   | 32      | 11         | 17   | 60    |

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
| `badlogic__pi-mono`  | `c`          | 14       |
| `badlogic__pi-mono`  | `html`       | 6        |
| `badlogic__pi-mono`  | `plaintext`  | 48       |
| `badlogic__pi-mono`  | `python`     | 7        |
| `badlogic__pi-mono`  | `yaml`       | 32       |
| `django__django`     | `toml`       | 9        |
| `rbtr__rbtr`         | `css`        | 2        |
| `rbtr__rbtr`         | `javascript` | 1        |
| `rbtr__rbtr`         | `plaintext`  | 48       |
| `rbtr__rbtr`         | `rust`       | 1        |
| `rbtr__rbtr`         | `toml`       | 45       |
| `rbtr__rbtr`         | `yaml`       | 13       |

## Classification — `provenance` × `query_kind`

How each generation strategy's text classifies as a request shape
(row-normalised). Provenance and query_kind are different axes: the
scatter here is that difference made visible.

| provenance  | concept | identifier | code  | n    |
| ----------- | ------- | ---------- | ----- | ---- |
| `body`      | 2.9%    | 40.1%      | 57.0% | 967  |
| `concept`   | 98.6%   | 1.3%       | 0.1%  | 1558 |
| `docstring` | 5.9%    | 88.7%      | 5.4%  | 389  |
| `name`      | 1.1%    | 97.7%      | 1.1%  | 709  |

## Examples

Sampled queries per provenance — the actual text fed to search,
verbatim.

**`body` → code** · python · `method` · __str__

````text
def __str__(self) -> str:
        return (self.family + "_" + self.variant) if self.variant else self.family
````

**`body` → code** · rst · `doc_section` ·

````text
[package.optional-dependencies]
grpc = [
    { name = "grpcio" },
]
````

**`body` → code** · rust · `class` · MarkerOperator

````text
impl Display for MarkerOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
````

**`concept` → concept** · rust · `class` · IntoIter

````text
iterate over flat dependency groups by name
````

**`concept` → concept** · css · `variable` · --white

````text
what color value does the white CSS variable define
````

**`concept` → concept** · css · `class` · .toclink

````text
reset the color of table of contents anchor links
````

**`docstring` → code** · rust · `class` · InstalledVersion

````text
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
````

**`docstring` → identifier** · python · `variable` · USE_UV_EXECUTABLE

````text
# Use the `uv build-backend` command rather than `uv-build`.
````

**`docstring` → identifier** · rust · `function` · collect_build_hints

````text
/// Collect hints from a build [`Error`] by inspecting its inner types.
````

**`name` → identifier** · python · `method` · interpreter

````text
ELFFile::interpreter
````

**`name` → identifier** · rust · `function` · find_python_from_active_python

````text
tests::find_python_from_active_python
````

**`name` → identifier** · rust · `method` · get

````text
TextStoreMode::get
````

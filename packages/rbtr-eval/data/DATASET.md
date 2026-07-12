# rbtr search-quality dataset

Characterisation of the query set **before** measurement — the input
to the benchmark, not its results. 2357 queries across
5 repos and 9 languages.

The three axes are independent: `symbol_kind` is the target chunk,
`provenance` is how the query was generated, and `query_kind` is
`classify_query(text)` — the request shape search routes on.

## Repos

The indexed commit and sampled sizes per repo.

| slug                 | sha            | symbols | sampled queries |
| -------------------- | -------------- | ------- | --------------- |
| `anthropics__skills` | `5128e1865d67` | 1913    | 264             |
| `astral-sh__uv`      | `cfe5277bc422` | 19007   | 200             |
| `badlogic__pi-mono`  | `a0a16c7762e6` | 9238    | 262             |
| `django__django`     | `e78a46a8fb29` | 59123   | 375             |
| `rbtr__rbtr`         | `d6ebe41d8953` | 2192    | 263             |

## Queries per language

| language   | n   |
| ---------- | --- |
| python     | 504 |
| markdown   | 490 |
| javascript | 358 |
| css        | 344 |
| typescript | 249 |
| rust       | 138 |
| rst        | 98  |
| sql        | 92  |
| html       | 84  |

## Target coverage — `symbol_kind` × `provenance`

Which kinds of chunk the queries target, and how each splits across
generation strategies. A kind absent here is not measured.

| symbol_kind | body | concept | docstring | name | total |
| ----------- | ---- | ------- | --------- | ---- | ----- |
| function    | 154  | 319     | 175       | 153  | 801   |
| doc_section | 175  | 322     | 0         | 175  | 672   |
| class       | 118  | 213     | 99        | 117  | 547   |
| method      | 78   | 139     | 40        | 80   | 337   |

## Target × request shape — `symbol_kind` × `query_kind`

For each target kind, the request shapes generated against it. Both
axes are independent of provenance.

| symbol_kind | concept | identifier | code | total |
| ----------- | ------- | ---------- | ---- | ----- |
| function    | 323     | 323        | 155  | 801   |
| doc_section | 327     | 269        | 76   | 672   |
| class       | 209     | 219        | 119  | 547   |
| method      | 140     | 128        | 69   | 337   |

## Classification — `provenance` × `query_kind`

How each generation strategy's text classifies as a request shape
(row-normalised). Provenance and query_kind are different axes: the
scatter here is that difference made visible.

| provenance  | concept | identifier | code  | n   |
| ----------- | ------- | ---------- | ----- | --- |
| `body`      | 1.5%    | 26.9%      | 71.6% | 525 |
| `concept`   | 97.8%   | 2.1%       | 0.1%  | 993 |
| `docstring` | 2.5%    | 91.1%      | 6.4%  | 314 |
| `name`      | 2.3%    | 93.5%      | 4.2%  | 525 |

## Examples

Sampled queries per provenance — the actual text fed to search,
verbatim.

**`body` → code** · rust · `method` · fmt

````text
fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Note that even though we have nested error types here, since we
        // don't expose them through std::error::Error::sour
````

**`body` → code** · css · `class` · .message-timestamp + .thinking-block .thinking-text,
…

````text
.message-timestamp + .thinking-block .thinking-text,
    .message-timestamp + .thinking-block .thinking-collapsed {
      padding-top: 0;
    }
````

**`body` → code** · javascript · `function` · _fd_write

````text
function _fd_write(fd,iov,iovcnt,pnum){try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);HEAPU32[pnum>>2]=num;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="Err
````

**`concept` → concept** · python · `function` · create_trusted_publisher

````text
Configure a crates.io trusted publishing GitHub integration for a crate
````

**`concept` → concept** · typescript · `function` · blockIndex

````text
get the index of the last block in a list
````

**`concept` → concept** · typescript · `class` · OAuthProviderInterface

````text
Interface defining an authentication provider's login flow, token refresh, and API key conversion
````

**`docstring` → identifier** · rust · `method` · from_workspace

````text
/// Lower the `build-system.requires` field from a `pyproject.toml` file.
````

**`docstring` → identifier** · rust · `class` · ResolvedRequirements

````text
/// Instantiate a [`ResolvedRequirements`] with the given [`Resolution`] and [`HashStrategy`].
````

**`docstring` → identifier** · css · `class` · .model-change

````text
/* Model change */
````

**`name` → identifier** · rust · `class` · Err

````text
VersionSpecifier::Err
````

**`name` → identifier** · css · `class` · .message-timestamp + .thinking-block
.thinking-text, …

````text
.message-timestamp + .thinking-block .thinking-text,
    .message-timestamp + .thinking-block .thinking-collapsed
````

**`name` → identifier** · javascript · `function` · _fd_write

````text
_fd_write
````

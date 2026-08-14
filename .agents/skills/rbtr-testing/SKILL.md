---
name: rbtr-testing
description: >-
  Testing conventions for rbtr: TDD workflow, fixture design,
  data-first test structure, and pytest patterns. Use when
  writing, modifying, or reviewing test files. Also trigger
  when you see pytest imports, test functions, or fixture
  definitions in the file being edited.
user-invocable: false
---

# rbtr testing conventions

## Workflow

**Red/green TDD.** Write a failing test first, then write the
code to make it pass. Run `just check` after each step.

**Read the red.** A new test must fail *for the reason it
describes* — read the assertion output, don't just note that it
failed. A test that **passes** when first written is a suspect
fixture, not a success: the setup probably never reaches the
state the bug needs.

**Committing a test whose fix comes later.** Mark it
`@pytest.mark.xfail(reason="…", strict=True)`. The suite stays
green, the bug is documented where it will be fixed, and
`strict` makes the test announce itself by *failing* the moment
the behaviour is right — at which point the marker comes off.

## Make the fixture actually reproduce the bug

A fixture that describes the bug in prose but not in state
produces a test that passes and proves nothing. Two traps in
this codebase have cost hours each:

- **Cache and dedup bugs need two builds, or two commits.**
  `add_chunk` buffers, so within a single build the chunks of
  an earlier file are not yet committed and a gate that reads
  committed state cannot see them. Anything about "the second
  occurrence is skipped" must put the first occurrence in a
  *previous* build — which is also what a real repo does.
- **Ranking tests need candidate spread.** `_normalise_col`
  min-max normalises each signal and *returns 0.0 when every
  candidate ties*. A fixture with only the two chunks under
  comparison therefore scores them both 0.0, and the order
  falls to the id tiebreak — the assertion passes or fails by
  hash. Include a weaker match so the distribution has a range.

When a test's outcome could be explained by the fixture rather
than the code, fix the fixture first.

## What to test — conceptual actions, not units

Test the **high-level conceptual actions** (the use cases:
build, embed, search, gc, diff, watch) against realistic data.
Those tests exercise the domain kernel (`rbtr.domain` — models,
identity, tokenisation) *transitively*, so pure domain code does
**not** need its own unit tests when a conceptual-action test
already covers it. A test earns its place by pinning a
behaviour a caller relies on, not by re-asserting an
encoding the use-case tests already prove.

- **Don't unit-test the kernel for its own sake.** A dedicated
  test of a pure domain function is warranted only for logic a
  use-case test can't naturally reach, or a specific edge case
  it can't isolate.
- **Test through real infrastructure, not fakes.** Prefer a real
  DuckDB store / pygit2 repo / daemon over a mock or a
  hand-rolled fake double. This is a deliberate trade: slightly
  slower tests that actually encode the conceptual action, over
  fast tests of an abstraction that only exists to be faked.
  Don't introduce a seam (a `Protocol`, an injected fake) whose
  only purpose is isolated unit testing.
- This is the *no-overlapping-tests* rule applied across layers:
  if the build test already proves `make_chunk_id` behaves, a
  separate `make_chunk_id` unit test is overlap.

## Test structure

- **Plain test functions only.** No test classes.
- **Parametrise over repetition.** One behaviour per test
  function, not one input value.
- **Data-first design.** Build tests around realistic shared
  datasets. Verify behaviours against concrete data, not
  anonymous stubs.
- **Bias against abstraction.** Prefer explicit, data-first
  scenarios even when they repeat a few lines. Helpers and
  parametrisation are tools, not goals.

## Cases over tests

- **Add cases, not tests.** When extending coverage for a
  new scenario, add a case function to an existing test
  before reaching for a new `test_*` function. A new test
  is warranted only when the *assertion logic* is genuinely
  different — not just the data.
- **No overlapping tests.** Before writing anything, check
  whether an existing test already covers the same behaviour.
  Two test functions must never exercise the same code path
  with different data — that is what cases are for.
- **Modify existing tests with care.** Do not rewrite or
  restructure a working test to accommodate new coverage.
  Prefer appending a case function that feeds the existing
  test.

## When to use pytest-cases vs `@pytest.mark.parametrize`

Use `pytest-cases` (`@case`, `@parametrize_with_cases`) for
behavioural scenarios — anything involving setup, shared
fixtures, complex data construction, or tagged families.

Plain `@pytest.mark.parametrize` is fine for pure expression
lookup tables: a flat list of `(input, expected_output)` tuples
with no setup or teardown. If the data outgrows a readable
tuple list, promote to cases.

## Mocking

- **No `unittest.mock`.** Use `pytest-mock` (`mocker` fixture).
- **Don't mock internal async iteration protocols.** Test
  through public APIs or mock at the LLM boundary.
- **Pure private functions can be tested directly** when doing
  so avoids mocking the full LLM chain.

## Fixtures

**Search before you write.** Grep the conftests up the tree
before adding a fixture — `store` (in-memory writable
`IndexStore`), `git_repo`, `isolated_db`, `two_commits` and
friends already exist, and a new fixture that constructs what
one of them provides also duplicates its teardown. Check
`rbtr.domain.models` before inventing a test-only type to carry
two values: `SnapshotRef` already pairs a repo with a snapshot,
and using it removes a hardcoded `repo_id=1` at every call site.

**Setup is proportional to assertions.** If fixture lines
outnumber assertion lines several times over, either the setup
is doing too much or the file is testing too little. Treat the
ratio as a review signal on the fixture, not on the tests.

**Return the thing under test.** A fixture returning
`tuple[Store, str]` forces every test to unpack positionally and
lets two same-typed values swap silently. Prefer: return the one
object; derive anything else in a second fixture that takes the
first. Reach for a frozen dataclass only when a *scenario
description* is being passed to a case family (see
`cases_common.py`), not to bundle a fixture's return.

**When a fixture draws criticism, subtract before you add.** The
usual fix for an awkward fixture is deleting part of it —
replacing a closure with structure, or a tuple with three new
types, makes it worse while appearing to address the note.

- **Fixtures over helpers.** Shared setup belongs in
  `@pytest.fixture`, not in loose helper functions — and not in
  a function nested inside a fixture, which hides its captured
  state. If a fixture body is too long, decompose into *smaller
  fixtures*.
- **Prefer independent fixtures to factory fixtures.** Factories
  hide the dependency graph. Reach for one only when the test
  truly needs many instances parametrised by caller-supplied
  arguments.
- **Helpers belong in cases or conftest.** If a helper
  function is needed to build test data, call it from case
  functions or define it in `conftest.py`. Don't scatter
  data-building helpers in test files.

### Test data and setup live in fixtures

1. **No module-level constants used by tests.** Even pure
   frozen values belong in fixtures.
2. **No module-level setup functions.** Long fixture bodies
   decompose into smaller fixtures, never into helpers.
3. **Pure projections are allowed.** Small module-level
   functions that transform values the test *already has*
   (e.g. `rank(results, chunk_id) -> int`) are fine — they
   don't hide dependencies.
4. **Shared values go in `conftest.py` fixtures.**
5. **Exception:** values the *production code* also uses
   (e.g. `SCHEMA_VERSION` imported for assertion) are not
   test data.
6. **Parametrize-time values may be module-level.** Pytest
   needs them at collection time (before fixtures run).
   Prefer inline literals; use module-level only when the
   data is large enough to hurt readability.

## Multi-line source fixtures

Use triple-quoted strings so the fixture reads like the file
it models:

```python
src = """\
def f():
    return 1
"""
```

Never use escaped newlines (`"a\nb\n"`) for multi-line source.
Implicit concatenation of escaped-newline strings is banned.

## Case file naming

Case files are named `cases_<module>.py` (plural). This
matches the `pytest-cases` auto-discovery convention —
`@parametrize_with_cases` finds `cases_<module>.py`
automatically without an explicit `cases=` argument.
Shared dataclass definitions used across case families
live in `cases_common.py`.

## Assertion helpers

Generic, reusable assertion helpers are allowed when they
eliminate repeated boilerplate and produce clear failure
messages. Keep them few — a handful per test directory,
not dozens.

- **Location:** in the test file when used only there,
  or in an `asserts.py` beside `conftest.py` when shared
  across test files in the same directory.
- **Naming:** `assert_*` prefix so intent is obvious.
- **Scope:** assertion helpers only. Data-building helpers
  belong in fixtures or case functions, not in test files
  or `asserts.py`.
- **Projections → assertions:** when a projection is only
  ever wrapped in the same assert pattern, promote it to
  an `assert_*` helper. The projection becomes a private
  implementation detail.

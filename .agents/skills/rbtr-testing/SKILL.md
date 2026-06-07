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

- **Fixtures over helpers.** Shared setup belongs in
  `@pytest.fixture`, not in loose helper functions. If a
  fixture body is too long, decompose into *smaller fixtures*.
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

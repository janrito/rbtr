# Review guidelines

Help the reviewer work through the changes methodically. Start
by reading tests, the PR description, and commit messages — they
reveal the author's intent faster than the implementation. Use
this reading to frame the review: what problem is being solved,
and does the code actually solve it?

## Design

- Does the change fit the surrounding architecture? Does it
  strengthen existing patterns or introduce a conflicting one?
- Are responsibilities clearly separated? Each unit should own
  its behaviour rather than exposing internals for callers to
  orchestrate.
- Where logic is repeated, is the duplication worth extracting?
  Shared code should only be factored out when the pattern is
  clear — premature abstraction is worse than a little
  repetition.
- Is the code solving a known present problem, or anticipating a
  future one that may never arrive?

Design issues are typically **blockers** when they introduce
architectural conflicts, **suggestions** otherwise.

## Correctness

- **Boundary conditions.** Look for off-by-one errors, empty
  collections, nil/null/undefined values, and behaviour at the
  extremes of valid input ranges.
- **Failure paths.** Are errors handled with specific types and
  helpful messages? Watch for catch blocks that log and continue,
  silently swallowed exceptions, and generic error types that
  obscure root causes.
- **Concurrency and ordering.** If the code runs in a concurrent
  context, are shared state mutations safe? Are async operations
  awaited or sequenced correctly? Look for race conditions
  between read and write.
- **State management.** Are mutations atomic where they need to
  be? Could a partial failure leave the system in an
  inconsistent state?
- **Scale.** What happens when input grows by 10× or 100×?
  Watch for unbounded queries, missing pagination, eager loads of
  large associations, and loops that make network calls per item.

Correctness issues are typically **blockers**.

## Readability

Readability is about whether someone unfamiliar with this code
can follow it quickly:

- Is the control flow clear? Straightforward step-by-step logic
  tends to be easier to work with than dense one-liners or deeply
  nested conditionals.
- Are literal values given meaningful names, or are there magic
  numbers scattered through the logic?
- Do comments explain _why_ a decision was made, not _what_ the
  code does? Flag any dead or commented-out code.

Readability issues are typically **suggestions**, occasionally
**nits**.

## Expressiveness

Expressiveness is about whether the code communicates its intent
through the domain:

- Do names reflect the domain concepts they represent? Vague
  names like `data`, `flag`, or `tmp` often mean there is room
  to express what the thing actually is.
- Do function signatures describe their contract? A function
  called `process` that takes `items` tells you nothing; one
  called `validate_order_line_items` tells you everything.
- Are abstractions named for what they _do_, not how they are
  _implemented_? Prefer `rate_limiter` over `token_bucket` unless
  the implementation is the point.

Expressiveness issues are typically **suggestions**.

## Testing

Tests are your primary source of intent. Read them first to
understand what the author believes the code should do.

- **Coverage of new behaviour.** Is the new or changed
  functionality tested? Where tests are missing, flag it
  explicitly — untested behaviour is open to interpretation and
  will quietly break.
- **Test quality.** Are tests verifying outcomes or
  implementation details? Tests coupled to internal structure
  break on refactors and provide false confidence.
- **Clarity of scenarios.** Do test names describe the scenario
  under test? Is the setup so complex that it obscures what is
  being verified?
- **Failure diagnostics.** When a test fails, will the output
  tell you what went wrong? Bare assertions without messages
  make debugging harder than it needs to be.

Missing tests for new behaviour are typically **blockers**.
Test quality and clarity issues are typically **suggestions**.

## Security

- **Input validation.** Is user-supplied or external input
  validated, sanitised, or escaped before use? Look for SQL
  injection, XSS, path traversal, and command injection vectors.
- **Authentication and authorisation boundaries.** Does the
  change respect existing access control? Are new endpoints or
  data paths protected appropriately?
- **Secrets and credentials.** Are there hardcoded tokens, keys,
  or passwords? Are secrets accessed through configuration or
  environment, not source?
- **Dependency risk.** Does the change introduce new
  dependencies? Are they well-maintained, appropriately scoped,
  and from trusted sources?

Security issues are typically **blockers**.

## Performance and data handling

- **Query patterns.** Watch for N+1 queries, missing indices on
  new query paths, and queries that fetch significantly more data
  than is used.
- **Memory and payload size.** Is large data loaded into memory
  when it could be streamed or paginated? Are API responses
  unbounded?
- **Caching.** If the change introduces or modifies caching, are
  invalidation conditions correct? Is there a risk of serving
  stale data?
- **Migrations and data changes.** Are database migrations
  reversible? Could they lock tables in production? Is there a
  backfill strategy if needed?

Performance issues range from **blocker** (missing indices on
high-traffic paths, table-locking migrations) to **suggestion**
(caching opportunities, payload optimisation).

## Producing feedback

When the reviewer is ready to leave a comment for the author,
help draft it using the author-facing voice

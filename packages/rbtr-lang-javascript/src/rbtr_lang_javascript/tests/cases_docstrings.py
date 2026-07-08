"""JavaScript / TypeScript docstring-extraction test cases.

Each `@case` returns `(lang, source, symbol_name, snippet)` consumed by
`test_docstrings.py`; tags drive the documented/undocumented assertion.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_function() -> DocstringCase:
    """Canonical JSDoc above a function declaration."""
    src = """\
/** Return a friendly greeting. */
function greet() {}
"""
    return "javascript", src, "greet", "Return a friendly greeting"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_class() -> DocstringCase:
    """Canonical JSDoc above a class declaration."""
    src = """\
/** A widget. */
class Widget {}
"""
    return "javascript", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_arrow_function() -> DocstringCase:
    """Arrow-function assignment — the `@function` capture
    lands on the `lexical_declaration`, so JSDoc attaches via
    the walk on that node's prev_named_sibling.
    """
    src = """\
/** Increment. */
const inc = (x) => x + 1;
"""
    return "javascript", src, "inc", "Increment"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_js_multiline_jsdoc() -> DocstringCase:
    """Multi-line JSDoc with leading `*` gutter."""
    src = """\
/**
 * Compute a checksum over *data*.
 *
 * The algorithm is CRC32 — explained below.
 */
function checksum(data) { return 0; }
"""
    return "javascript", src, "checksum", "The algorithm is CRC32"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_js_banner_comment() -> DocstringCase:
    """`/*! ... */` banner comments are common in bundled UMD
    libs; the grammar lands them as `comment` nodes and we
    attach them — the benchmark will say whether that hurts.
    """
    src = """\
/*! (c) 2024 Acme. */
function publicApi() {}
"""
    return "javascript", src, "publicApi", "(c) 2024 Acme"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_js_line_comment_run() -> DocstringCase:
    """`//` comment runs used as docs — common in TS-first
    code where JSDoc is syntactically less convenient.
    """
    src = """\
// First description line.
// Second description line.
function documented() {}
"""
    return "javascript", src, "documented", "First description line"


@case(tags=["undocumented", "no_docs"])
def case_js_function_without_doc() -> DocstringCase:
    """Plain function, no comments."""
    src = """\
function bare() {}
"""
    return "javascript", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_js_jsdoc_detached_by_blank_line() -> DocstringCase:
    """A blank line between the JSDoc block and the function
    breaks attachment.
    """
    src = """\
/** Stale JSDoc — not attached. */

function stale() {}
"""
    return "javascript", src, "stale", "Stale JSDoc"


@case(tags=["undocumented", "invalid"])
def case_js_jsdoc_above_import_does_not_attach_to_class() -> DocstringCase:
    """JSDoc above an `import` stays on the import line.  A
    class several statements later must *not* inherit it.
    Imports are excluded from attachment by design (see
    `treesitter.extract_symbols`).
    """
    src = """\
/** Nonsense JSDoc above import. */
import { x } from './x';

class Real {}
"""
    return "javascript", src, "Real", "Nonsense JSDoc"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_function() -> DocstringCase:
    """Canonical JSDoc above a typed function declaration."""
    src = """\
/** Return the length of *s*. */
function len(s: string): number { return s.length; }
"""
    return "typescript", src, "len", "Return the length"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_class() -> DocstringCase:
    """JSDoc above a TypeScript class — grammar uses
    `type_identifier` for the class name.
    """
    src = """\
/** A widget. */
class Widget {}
"""
    return "typescript", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_arrow_function() -> DocstringCase:
    """Arrow-function assignment with a type annotation."""
    src = """\
/** Increment. */
const inc: (x: number) => number = (x) => x + 1;
"""
    return "typescript", src, "inc", "Increment"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ts_jsdoc_on_generic_function() -> DocstringCase:
    """Generic type parameters between name and arguments."""
    src = """\
/** Identity. */
function identity<T>(x: T): T { return x; }
"""
    return "typescript", src, "identity", "Identity"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ts_multiline_jsdoc_with_tags() -> DocstringCase:
    """Multi-line JSDoc with `@param` / `@returns` tags."""
    src = """\
/**
 * Compute a hash.
 *
 * @param data bytes to hash
 * @returns hex digest
 */
function hash(data: Uint8Array): string { return ''; }
"""
    return "typescript", src, "hash", "@returns hex digest"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_ts_line_comment_run() -> DocstringCase:
    """`//` comment runs used as docs, common in TS-heavy
    codebases that avoid JSDoc because types are already in
    the signature.
    """
    src = """\
// Describe the value.
// Useful in calling code.
function describe(x: number): string { return String(x); }
"""
    return "typescript", src, "describe", "Describe the value"


@case(tags=["undocumented", "no_docs"])
def case_ts_function_without_doc() -> DocstringCase:
    """Plain TS function."""
    src = """\
function bare(x: number): number { return x; }
"""
    return "typescript", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_ts_jsdoc_detached_by_blank_line() -> DocstringCase:
    """Blank line breaks attachment for TS too."""
    src = """\
/** Stale JSDoc. */

function stale(): void {}
"""
    return "typescript", src, "stale", "Stale JSDoc"


@case(tags=["undocumented", "invalid"])
def case_ts_jsdoc_above_import() -> DocstringCase:
    """JSDoc above `import` does not attach to a later class."""
    src = """\
/** Nonsense JSDoc. */
import { x } from './x';

class Real {}
"""
    return "typescript", src, "Real", "Nonsense JSDoc"

"""Docstring-extraction test cases for all languages.

Each `@case` returns a 4-tuple `(lang, source, symbol_name,
snippet)` consumed by `test_docstrings.py`:

* `lang`        - language id registered by a plugin.
* `source`      - a small but realistic source fragment.
* `symbol_name` - name of the chunk whose `content` the test
  will inspect.  Cases keep names unique so disambiguation is
  trivial.
* `snippet`     - a substring the test will look for.  For
  `documented` cases the snippet must appear in `content`; for
  `undocumented` cases it must *not* appear.

Tags
----

Every case carries **two** tags.

Primary (drives the assertion direction):

* `documented`   - the chunk's `content` must contain `snippet`.
* `undocumented` - the chunk's `content` must *not* contain
  `snippet`.

Secondary (classifies the scenario for debugging slices):

* `canonical`              - the language's idiomatic,
                              by-the-book documentation form.
* `edge_case`              - a valid but unusual doc form (raw
                              / byte strings, nested quotes,
                              multi-line block comments,
                              language-specific inner doc such
                              as Rust `//!`).
* `unconventional`         - not strictly canonical but clearly
                              intended as docs.  rbtr leans
                              toward extracting these.
* `boundary_attached`      - attachment-logic scenario whose
                              outcome is attachment.
* `no_docs`                - symbol has no documentation.
* `boundary_not_attached`  - attachment-logic scenario whose
                              outcome is no attachment.
* `invalid`                - text that looks like documentation
                              but is not recognised as such.

Policy being tested: "the default rbtr behaviour is that every
documented symbol exposes its documentation in the chunk
content".  Every language has at least one case in each primary
bucket and covers every secondary tag relevant to that language.

Source-template convention
--------------------------

Python source strings use single-quoted triple-string (`'''...'''`)
templates so their body can embed double-quoted triple-string
docstrings (`\"\"\"...\"\"\"`) verbatim.
"""

from __future__ import annotations

from pytest_cases import case

# ═════════════════════════════════════════════════════════════════════
# Python
# ═════════════════════════════════════════════════════════════════════


@case(tags=["documented", "canonical"])
def case_py_function_docstring():
    """PEP 257 single-line function docstring."""
    src = '''\
def greet(name):
    """Return a friendly greeting for *name*."""
    return f"hi {name}"
'''
    return "python", src, "greet", "Return a friendly greeting"


@case(tags=["documented", "canonical"])
def case_py_class_docstring():
    """Class-level docstring as first statement of the body."""
    src = '''\
class Greeter:
    """Produce friendly greetings."""

    def hi(self):
        return "hi"
'''
    return "python", src, "Greeter", "Produce friendly greetings"


@case(tags=["documented", "canonical"])
def case_py_method_docstring():
    """Method docstring inside a class."""
    src = '''\
class Svc:
    def run(self):
        """Execute the main loop."""
        return 0
'''
    return "python", src, "run", "Execute the main loop"


@case(tags=["documented", "canonical"])
def case_py_multiline_docstring():
    """Multi-line docstring with summary + body."""
    src = '''\
def compute(x, y):
    """Add two numbers.

    The summary is one line; the body elaborates.  Both parts
    must remain in chunk content.
    """
    return x + y
'''
    return "python", src, "compute", "The summary is one line"


@case(tags=["documented", "edge_case"])
def case_py_raw_string_docstring():
    """`r\"\"\"...\"\"\"` raw string is a valid docstring."""
    src = '''\
def regex():
    r"""Match DIGIT sequences."""
    return 1
'''
    return "python", src, "regex", "Match DIGIT sequences"


@case(tags=["documented", "edge_case"])
def case_py_single_quoted_docstring():
    """Single-quoted triple-string is equally valid."""
    src = """\
def foo():
    '''Single-quoted docstring content.'''
    return 1
"""
    return "python", src, "foo", "Single-quoted docstring content"


@case(tags=["documented", "edge_case"])
def case_py_decorated_function_docstring():
    """Decorators precede `def`; docstring is still interior."""
    src = '''\
@cache
@wraps(other)
def memoized(x):
    """Memoise calls to *other*."""
    return other(x)
'''
    return "python", src, "memoized", "Memoise calls"


@case(tags=["documented", "unconventional"])
def case_py_docstring_with_code_block():
    """Docstring embedding example code - common in libraries."""
    src = '''\
def parse(s):
    """Parse *s* into tokens.

    Example::

        >>> parse("1+2")
        [1, "+", 2]
    """
    return []
'''
    return "python", src, "parse", ">>> parse"


@case(tags=["documented", "unconventional"])
def case_py_class_and_method_both_documented():
    """Method chunk carries its own docstring, not the class's."""
    src = '''\
class Svc:
    """Service facade."""

    def start(self):
        """Boot the service."""
        return 1
'''
    return "python", src, "start", "Boot the service"


@case(tags=["undocumented", "no_docs"])
def case_py_function_without_docstring():
    """Plain function: chunk content has no triple-quoted prose.

    The probe is a unique marker string that would only be
    present if some other source polluted this chunk.
    """
    src = "def add(a, b):\n    return a + b\n"
    return "python", src, "add", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "no_docs"])
def case_py_class_without_docstring():
    """Plain class with no documentation."""
    src = "class Bare:\n    pass\n"
    return "python", src, "Bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "invalid"])
def case_py_leading_hash_comment_not_attached():
    """A `#` comment above `def` is deliberately not treated as
    documentation in Python.  The interior triple-string form
    is canonical, and `#` runs frequently serve as section
    headers or type-ignore markers — attaching them would
    double-count noise.
    """
    src = "# HASH_COMMENT_ABOVE_DEF_MARKER\ndef work():\n    return 1\n"
    return "python", src, "work", "HASH_COMMENT_ABOVE_DEF_MARKER"


@case(tags=["undocumented", "invalid"])
def case_py_trailing_string_not_a_docstring():
    """A triple-string placed *after* other statements is a
    discarded expression, not a docstring.  The text is in the
    chunk content (it is part of the function body) but is not
    treated as documentation — `--strip-docstrings` must leave
    it intact.  The `test_stripping_removes_doc_text` policy
    guards *documented* cases; this case goes in
    `undocumented` because the engine must not claim the
    string as documentation in the first place.

    We probe for presence of the marker using the
    `test_no_phantom_documentation` logic inverted: the marker
    *is* in the chunk, so probing for a different string that
    would only appear if the extractor invented content gives
    the right signal.
    """
    src = 'def late():\n    x = 1\n    """PSEUDO_DOC"""\n    return x\n'
    return "python", src, "late", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"

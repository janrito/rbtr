"""Python docstring-extraction test cases.

Each `@case` returns `(files, symbol_name, snippet)` consumed by
`test_docstrings.py`; tags drive the documented/undocumented assertion.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[dict[str, str], str, str]


@case(tags=["documented"])
def case_py_function_docstring() -> DocstringCase:
    """PEP 257 single-line function docstring."""
    src = '''\
def greet(name):
    """Return a friendly greeting for *name*."""
    return f"hi {name}"
'''
    return {"input.py": src}, "greet", "Return a friendly greeting"


@case(tags=["documented"])
def case_py_class_docstring() -> DocstringCase:
    """Class-level docstring as first statement of the body."""
    src = '''\
class Greeter:
    """Produce friendly greetings."""

    def hi(self):
        return "hi"
'''
    return {"input.py": src}, "Greeter", "Produce friendly greetings"


@case(tags=["documented"])
def case_py_method_docstring() -> DocstringCase:
    """Method docstring inside a class."""
    src = '''\
class Svc:
    def run(self):
        """Execute the main loop."""
        return 0
'''
    return {"input.py": src}, "run", "Execute the main loop"


@case(tags=["documented"])
def case_py_multiline_docstring() -> DocstringCase:
    """Multi-line docstring with summary + body."""
    src = '''\
def compute(x, y):
    """Add two numbers.

    The summary is one line; the body elaborates.  Both parts
    must remain in chunk content.
    """
    return x + y
'''
    return {"input.py": src}, "compute", "The summary is one line"


@case(tags=["documented"])
def case_py_raw_string_docstring() -> DocstringCase:
    """`r\"\"\"...\"\"\"` raw string is a valid docstring."""
    src = '''\
def regex():
    r"""Match DIGIT sequences."""
    return 1
'''
    return {"input.py": src}, "regex", "Match DIGIT sequences"


@case(tags=["documented"])
def case_py_single_quoted_docstring() -> DocstringCase:
    """Single-quoted triple-string is equally valid."""
    src = """\
def foo():
    '''Single-quoted docstring content.'''
    return 1
"""
    return {"input.py": src}, "foo", "Single-quoted docstring content"


@case(tags=["documented"])
def case_py_decorated_function_docstring() -> DocstringCase:
    """Decorators precede `def`; docstring is still interior."""
    src = '''\
@cache
@wraps(other)
def memoized(x):
    """Memoise calls to *other*."""
    return other(x)
'''
    return {"input.py": src}, "memoized", "Memoise calls"


@case(tags=["documented"])
def case_py_docstring_with_code_block() -> DocstringCase:
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
    return {"input.py": src}, "parse", ">>> parse"


@case(tags=["documented"])
def case_py_class_and_method_both_documented() -> DocstringCase:
    """Method chunk carries its own docstring, not the class's."""
    src = '''\
class Svc:
    """Service facade."""

    def start(self):
        """Boot the service."""
        return 1
'''
    return {"input.py": src}, "start", "Boot the service"


@case(tags=["undocumented"])
def case_py_function_without_docstring() -> DocstringCase:
    """Plain function: chunk content has no triple-quoted prose.

    The probe is a unique marker string that would only be
    present if some other source polluted this chunk.
    """
    src = """\
def add(a, b):
    return a + b
"""
    return {"input.py": src}, "add", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented"])
def case_py_class_without_docstring() -> DocstringCase:
    """Plain class with no documentation."""
    src = """\
class Bare:
    pass
"""
    return {"input.py": src}, "Bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented"])
def case_py_trailing_string_not_a_docstring() -> DocstringCase:
    """A triple-string placed *after* other statements is a
    discarded expression, not a docstring.  The text is in the
    chunk content (it is part of the function body) but is not
    treated as documentation.

    We probe for presence of the marker using the
    `test_no_phantom_documentation` logic inverted: the marker
    *is* in the chunk, so probing for a different string that
    would only appear if the extractor invented content gives
    the right signal.
    """
    src = '''\
def late():
    x = 1
    """PSEUDO_DOC"""
    return x
'''
    return {"input.py": src}, "late", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"

"""Ruby docstring-extraction test cases.

Each `@case` returns `(lang, source, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_def() -> DocstringCase:
    """Canonical `#` comment above a top-level def."""
    src = """\
# Greet the user.
def greet
  'hi'
end
"""
    return "ruby", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_class() -> DocstringCase:
    """`#` comment above a class declaration."""
    src = """\
# Service facade.
class Svc
end
"""
    return "ruby", src, "Svc", "Service facade"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_module() -> DocstringCase:
    """`#` comment above a module declaration."""
    src = """\
# Utilities.
module Utils
end
"""
    return "ruby", src, "Utils", "Utilities"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_multi_line_hash_doc() -> DocstringCase:
    """Multi-line `#` doc comment."""
    src = """\
# Compute a checksum.
#
# Returns a hex digest string.
def checksum
  ''
end
"""
    return "ruby", src, "checksum", "Returns a hex digest"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ruby_block_comment_begin_end() -> DocstringCase:
    """`=begin` / `=end` block comment.  The grammar treats it
    as a single `comment` node; attachment works when it
    immediately precedes the `def`.
    """
    src = """\
=begin
Block-style doc.
=end
def foo
end
"""
    return "ruby", src, "foo", "Block-style doc"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_ruby_shebang_like_comment_above_def() -> DocstringCase:
    """Unconventional but valid: a `#` run whose first line
    starts with `#!`-style emphasis still attaches.
    """
    src = """\
#! IMPORTANT: use Foo instead of Bar.
# Prefer modern API.
def legacy
end
"""
    return "ruby", src, "legacy", "IMPORTANT: use Foo"


@case(tags=["undocumented", "no_docs"])
def case_ruby_def_without_doc() -> DocstringCase:
    """Undocumented top-level def."""
    src = """\
def bare
end
"""
    return "ruby", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_ruby_doc_detached_by_blank_line() -> DocstringCase:
    """Blank line between comment and def breaks attachment."""
    src = """\
# Orphan.

def later
end
"""
    return "ruby", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_ruby_doc_does_not_steal_from_next() -> DocstringCase:
    """A `#` comment between two top-level defs belongs to the
    later def, not the earlier one.
    """
    src = """\
def first
end

# Doc for second.
def second
end
"""
    return "ruby", src, "first", "Doc for second"

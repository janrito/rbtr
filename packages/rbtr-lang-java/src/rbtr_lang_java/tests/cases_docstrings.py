"""Java docstring-extraction test cases.

Each `@case` returns `(files, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[dict[str, str], str, str]


@case(tags=["documented"])
def case_java_javadoc_on_class() -> DocstringCase:
    """Canonical Javadoc above a class declaration."""
    src = """\
/** A widget. */
public class Widget {}
"""
    return {"input.java": src}, "Widget", "A widget"


@case(tags=["documented"])
def case_java_javadoc_on_method() -> DocstringCase:
    """Canonical Javadoc above a method declaration."""
    src = """\
public class Widget {
    /** Render the widget. */
    public void render() {}
}
"""
    return {"input.java": src}, "render", "Render the widget"


@case(tags=["documented"])
def case_java_multi_line_javadoc_with_tags() -> DocstringCase:
    """Multi-line Javadoc with `@param` / `@return`."""
    src = """\
public class Widget {
    /**
     * Compute a hash.
     * @param data input
     * @return hex string
     */
    public String hash(byte[] data) { return ""; }
}
"""
    return {"input.java": src}, "hash", "@return hex string"


@case(tags=["documented"])
def case_java_javadoc_above_annotated_method() -> DocstringCase:
    """Annotations parse as part of the method node's
    `modifiers` child, so the Javadoc remains the method's
    `prev_named_sibling` and attaches correctly.
    """
    src = """\
public class Widget {
    /** Deprecated API. */
    @Deprecated
    public void old() {}
}
"""
    return {"input.java": src}, "old", "Deprecated API"


@case(tags=["documented"])
def case_java_line_comment_run_on_method() -> DocstringCase:
    """Plain `//` comment run attaches too."""
    src = """\
public class C {
    // A line comment.
    // Second line.
    public void m() {}
}
"""
    return {"input.java": src}, "m", "A line comment"


@case(tags=["undocumented"])
def case_java_method_without_doc() -> DocstringCase:
    """Undocumented method."""
    src = """\
public class C { public void bare() {} }
"""
    return {"input.java": src}, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented"])
def case_java_javadoc_detached_by_blank_line() -> DocstringCase:
    """Blank line between Javadoc and method breaks attachment."""
    src = """\
public class C {
    /** Orphan. */

    public void later() {}
}
"""
    return {"input.java": src}, "later", "Orphan"


@case(tags=["undocumented"])
def case_java_javadoc_on_previous_method_does_not_attach_to_later() -> DocstringCase:
    """A Javadoc above method A is not attached to later method B."""
    src = """\
public class C {
    /** Doc for first. */
    public void first() {}

    public void second() {}
}
"""
    return {"input.java": src}, "second", "Doc for first"

"""HTML extraction test cases."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_html_semantic_elements() -> SymbolCase:
    """HTML names an element by its id (even beside a class), else its tag."""
    src = """\
<html>
<body>
  <main id="content"><p>Hello</p></main>
  <article class="post" id="first">Text</article>
  <nav>Links</nav>
</body>
</html>
"""
    return (
        {"input.html": src},
        [
            ("doc_section", "body", ""),
            ("doc_section", "content", ""),
            ("doc_section", "first", ""),
            ("doc_section", "nav", ""),
        ],
    )


@case(tags=["symbol"])
def case_html_anchors_by_id_and_by_a_name() -> SymbolCase:
    """An element is named by its `id`, an `<a>` by its `name` too; a form field's `name` names nothing."""
    src = """\
<p>
  <input type="text" id="searchbar" name="q">
  <a name="legacy">old</a>
  <select name="choice"></select>
</p>
"""
    return (
        {"input.html": src},
        [("doc_section", "searchbar", ""), ("doc_section", "legacy", "")],
    )

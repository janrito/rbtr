"""HTML extraction test cases."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


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
        "html",
        src,
        [
            ("doc_section", "body", ""),
            ("doc_section", "content", ""),
            ("doc_section", "first", ""),
            ("doc_section", "nav", ""),
        ],
    )

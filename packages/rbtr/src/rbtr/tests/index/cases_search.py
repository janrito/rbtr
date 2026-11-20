"""Cases for store-level search behaviours.

Each case returns a `SearchScenario`: the files a snapshot holds, a
query, and the names the hits should carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_cases import case


@dataclass(frozen=True)
class SearchScenario:
    """Files to build + query + the names the hits should carry."""

    files: dict[str, bytes]
    query: str
    expected_hit_names: list[str] = field(default_factory=list)


@case(tags=["fts_hit"])
def case_fts_content_keyword() -> SearchScenario:
    """FTS finds a chunk by a keyword in its content."""
    a = b"""\
def greet():
    print('hello')
"""
    return SearchScenario(files={"a.py": a}, query="hello", expected_hit_names=["greet"])


@case(tags=["fts_hit"])
def case_fts_camel_case_split() -> SearchScenario:
    """FTS splits CamelCase and finds by component."""
    a = b"""\
class AgentDeps:
    pass
"""
    return SearchScenario(files={"a.py": a}, query="agent", expected_hit_names=["AgentDeps"])


@case(tags=["fts_hit"])
def case_fts_snake_case_split() -> SearchScenario:
    """FTS splits snake_case and finds by component."""
    a = b"""\
def parse_json_response():
    pass
"""
    return SearchScenario(
        files={"a.py": a}, query="json", expected_hit_names=["parse_json_response"]
    )


@case(tags=["fts_empty"])
def case_fts_empty_query() -> SearchScenario:
    """Empty query returns no results."""
    a = b"""\
def a():
    pass
"""
    return SearchScenario(files={"a.py": a}, query="")


@case(tags=["fts_empty"])
def case_fts_nonsense_query() -> SearchScenario:
    """Nonsense query returns no results."""
    a = b"""\
def a():
    pass
"""
    return SearchScenario(files={"a.py": a}, query="xyzzyplugh")


@case(tags=["name_hit"])
def case_name_substring_match() -> SearchScenario:
    """Name search finds by substring."""
    a = b"""\
def calculate_standard_deviation():
    pass
"""
    return SearchScenario(
        files={"a.py": a}, query="standard", expected_hit_names=["calculate_standard_deviation"]
    )


@case(tags=["name_empty"])
def case_name_no_match() -> SearchScenario:
    """Name search returns empty for non-matching query."""
    a = b"""\
def helper():
    pass
"""
    return SearchScenario(files={"a.py": a}, query="xyzzy")


@case(tags=["unified_no_embed"])
def case_unified_search_without_embeddings() -> SearchScenario:
    """Unified search works when no embeddings exist.

    The semantic channel fails gracefully and redistributes
    its weight to FTS + name.
    """
    a = b"""\
class AppConfig:
    pass
"""
    b = b"""\
def load_config():
    pass
"""
    return SearchScenario(
        files={"a.py": a, "b.py": b},
        query="config",
        expected_hit_names=["AppConfig", "load_config"],
    )

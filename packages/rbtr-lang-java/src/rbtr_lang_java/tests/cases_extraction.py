"""Java extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_java_class() -> SymbolCase:
    """class User {}."""
    return "java", "class User {}\n", [("class", "User", "")]


@case(tags=["symbol"])
def case_java_public_class() -> SymbolCase:
    """public class App {}."""
    return "java", "public class App {}\n", [("class", "App", "")]


@case(tags=["symbol"])
def case_java_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return "java", "class Admin extends User {}\n", [("class", "Admin", "")]


@case(tags=["symbol"])
def case_java_class_implements() -> SymbolCase:
    """class UserService implements Service {}."""
    return "java", "class UserService implements Service {}\n", [("class", "UserService", "")]


@case(tags=["symbol"])
def case_java_multiple_classes() -> SymbolCase:
    """Multiple classes."""
    src = """\
class Foo {}
class Bar {}
"""
    return "java", src, [("class", "Foo", ""), ("class", "Bar", "")]


@case(tags=["symbol"])
def case_java_method_in_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Service {
    void process() {}
}
"""
    return "java", src, [("method", "process", "Service")]


@case(tags=["symbol"])
def case_java_multiple_methods() -> SymbolCase:
    """Multiple methods."""
    src = """\
class Svc {
    void start() {}
    void stop() {}
}
"""
    return "java", src, [("method", "start", "Svc"), ("method", "stop", "Svc")]


@case(tags=["symbol"])
def case_java_static_method() -> SymbolCase:
    """Static method still scoped."""
    src = """\
class Factory {
    static Object create() { return null; }
}
"""
    return "java", src, [("method", "create", "Factory")]


@case(tags=["symbol"])
def case_java_method_params() -> SymbolCase:
    """Method with parameters."""
    src = """\
class Calc {
    int add(int a, int b) { return a + b; }
}
"""
    return "java", src, [("method", "add", "Calc")]


@case(tags=["symbol"])
def case_java_nested_class() -> SymbolCase:
    """Nested class scoped to outer, method scoped to inner."""
    src = """\
class Outer {
    class Inner {
        void deep() {}
    }
}
"""
    return (
        "java",
        src,
        [("class", "Outer", ""), ("class", "Inner", "Outer"), ("method", "deep", "Outer::Inner")],
    )


@case(tags=["symbol"])
def case_java_triple_nested_class() -> SymbolCase:
    """Three levels of nested class compose the full path."""
    src = """\
class Outer {
    class Mid {
        class Inner {
            void deep() {}
        }
    }
}
"""
    return "java", src, [("method", "deep", "Outer::Mid::Inner")]


@case(tags=["import"])
def case_java_import_class() -> ImportCase:
    """import java.util.HashMap."""
    return "java", "import java.util.HashMap;\n", {"module": "java.util.HashMap"}


@case(tags=["import"])
def case_java_import_deeply_nested() -> ImportCase:
    """import com.example.app.models.User."""
    return (
        "java",
        "import com.example.app.models.User;\n",
        {"module": "com.example.app.models.User"},
    )


@case(tags=["import"])
def case_java_import_static() -> ImportCase:
    """import static org.junit.Assert.assertEquals."""
    return (
        "java",
        "import static org.junit.Assert.assertEquals;\n",
        {"module": "org.junit.Assert.assertEquals"},
    )


@case(tags=["import"])
def case_java_import_static_method() -> ImportCase:
    """import static java.util.Collections.sort."""
    return (
        "java",
        "import static java.util.Collections.sort;\n",
        {"module": "java.util.Collections.sort"},
    )


@case(tags=["multi_import"])
def case_java_multiple_imports() -> MultiImportCase:
    """Two import statements."""
    src = """\
import java.util.List;
import java.util.Map;
"""
    return (
        "java",
        src,
        2,
        [
            {"module": "java.util.List"},
            {"module": "java.util.Map"},
        ],
    )


@case(tags=["mixed"])
def case_java_full_class() -> MixedCase:
    """Realistic Java class with Javadoc on every member.

    Expected tuple unchanged; content assertions in
    `test_docstrings.py`.
    """
    src = """\
import java.util.List;
import java.util.ArrayList;

/** Tracks registered user names. */
public class UserService {
    private List<String> names;

    /** Append a new name to the registry. */
    public void addName(String name) {
        names.add(name);
    }

    /** Return the current list of names. */
    public List<String> getNames() {
        return names;
    }
}
"""
    return (
        "java",
        src,
        {"import", "class", "method"},
        [("addName", "UserService"), ("getNames", "UserService")],
    )

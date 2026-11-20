"""Go extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]
type ImportCase = tuple[dict[str, str], dict[str, str | list[dict[str, str]]]]
type MultiImportCase = tuple[dict[str, str], int, list[dict[str, str | list[dict[str, str]]]]]
type MixedCase = tuple[dict[str, str], set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_go_function() -> SymbolCase:
    """func hello() {}."""
    src = """\
package main

func hello() {}
"""
    return {"input.go": src}, [("function", "hello", "")]


@case(tags=["symbol"])
def case_go_function_params() -> SymbolCase:
    """func add(a int, b int) int."""
    src = """\
package main

func add(a int, b int) int { return a + b }
"""
    return {"input.go": src}, [("function", "add", "")]


@case(tags=["symbol"])
def case_go_multiple_functions() -> SymbolCase:
    """Multiple functions."""
    src = """\
package main

func foo() {}
func bar() {}
func baz() {}
"""
    return (
        {"input.go": src},
        [
            ("function", "foo", ""),
            ("function", "bar", ""),
            ("function", "baz", ""),
        ],
    )


@case(tags=["symbol"])
def case_go_method() -> SymbolCase:
    """Value receiver method."""
    src = """\
package main

type User struct{}

func (u User) Name() string { return u.name }
"""
    return {"input.go": src}, [("method", "Name", "User")]


@case(tags=["symbol"])
def case_go_pointer_method() -> SymbolCase:
    """Pointer receiver method."""
    src = """\
package main

type Svc struct{}

func (s *Svc) Start() {}
"""
    return {"input.go": src}, [("method", "Start", "Svc")]


@case(tags=["symbol"])
def case_go_struct() -> SymbolCase:
    """type User struct."""
    src = """\
package main

type User struct {
    Name string
}
"""
    return {"input.go": src}, [("class", "User", "")]


@case(tags=["symbol"])
def case_go_interface() -> SymbolCase:
    """type Reader interface."""
    src = """\
package main

type Reader interface {
    Read(p []byte) (int, error)
}
"""
    return {"input.go": src}, [("class", "Reader", "")]


@case(tags=["symbol"])
def case_go_type_alias() -> SymbolCase:
    """type ID string."""
    src = """\
package main

type ID string
"""
    return {"input.go": src}, [("class", "ID", "")]


@case(tags=["symbol"])
def case_go_multiple_types() -> SymbolCase:
    """Multiple type declarations."""
    src = """\
package main

type Foo struct{}
type Bar struct{}
"""
    return {"input.go": src}, [("class", "Foo", ""), ("class", "Bar", "")]


@case(tags=["import"])
def case_go_import_single() -> ImportCase:
    """import "fmt"."""
    src = """\
package main
import "fmt"
"""
    return {"input.go": src}, {"module": "fmt", "bindings": [{"name": "fmt", "local": "fmt"}]}


@case(tags=["import"])
def case_go_import_nested() -> ImportCase:
    """import "os/exec"."""
    src = """\
package main
import "os/exec"
"""
    return {"input.go": src}, {
        "module": "os/exec",
        "bindings": [{"name": "os/exec", "local": "exec"}],
    }


@case(tags=["import"])
def case_go_import_url() -> ImportCase:
    """import "github.com/user/repo"."""
    src = """\
package main
import "github.com/user/repo"
"""
    return (
        {"input.go": src},
        {
            "module": "github.com/user/repo",
            "bindings": [{"name": "github.com/user/repo", "local": "repo"}],
        },
    )


@case(tags=["import"])
def case_go_import_aliased() -> ImportCase:
    """import f "fmt" — the module, bound under its alias."""
    src = """\
package main
import f "fmt"
"""
    return {"input.go": src}, {"module": "fmt", "bindings": [{"name": "fmt", "local": "f"}]}


@case(tags=["import"])
def case_go_import_dot() -> ImportCase:
    """import . "strings" — every name of the package, written `*`."""
    src = """\
package main
import . "strings"
"""
    return {"input.go": src}, {"module": "strings", "names": "*"}


@case(tags=["import"])
def case_go_import_blank() -> ImportCase:
    """import _ "embed" — imported for its effect, binding nothing."""
    src = """\
package main
import _ "embed"
"""
    return {"input.go": src}, {"module": "embed"}


@case(tags=["multi_import"])
def case_go_import_grouped() -> MultiImportCase:
    """import ("fmt" "os") — separate chunks per spec."""
    src = """\
package main
import (
    "fmt"
    "os"
)
"""
    return (
        {"input.go": src},
        2,
        [
            {"module": "fmt", "bindings": [{"name": "fmt", "local": "fmt"}]},
            {"module": "os", "bindings": [{"name": "os", "local": "os"}]},
        ],
    )


@case(tags=["multi_import"])
def case_go_import_grouped_paths() -> MultiImportCase:
    """Grouped import with nested paths — separate chunks."""
    src = """\
package main
import (
    "fmt"
    "os/exec"
    "net/http"
)
"""
    return (
        {"input.go": src},
        3,
        [
            {"module": "fmt", "bindings": [{"name": "fmt", "local": "fmt"}]},
            {"module": "os/exec", "bindings": [{"name": "os/exec", "local": "exec"}]},
            {"module": "net/http", "bindings": [{"name": "net/http", "local": "http"}]},
        ],
    )


@case(tags=["import"])
def case_go_import_grouped_single() -> ImportCase:
    """Grouped import with one item."""
    src = """\
package main
import (
    "fmt"
)
"""
    return {"input.go": src}, {"module": "fmt", "bindings": [{"name": "fmt", "local": "fmt"}]}


@case(tags=["mixed"])
def case_go_full_module() -> MixedCase:
    """Realistic Go module with godoc-style comments.

    Expected-kinds tuple unchanged; content assertions are in
    `test_docstrings.py`.
    """
    src = """\
package main

import (
    "fmt"
    "os"
)

// Config is the runtime configuration for the service.
type Config struct {
    Name string
}

// String returns a human-readable summary of the Config.
func (c Config) String() string { return c.Name }

// main is the entry point for the service.
func main() {
    fmt.Println("hello")
}
"""
    return {"input.go": src}, {"import", "class", "method", "function"}, [("String", "Config")]


@case(tags=["symbol"])
def case_go_package_var() -> SymbolCase:
    """Package-level var."""
    src = """\
package main
var MaxSize = 100
"""
    return {"input.go": src}, [("variable", "MaxSize", "")]


@case(tags=["symbol"])
def case_go_package_const() -> SymbolCase:
    """Package-level const."""
    src = """\
package main
const Timeout = 30
"""
    return {"input.go": src}, [("variable", "Timeout", "")]


@case(tags=["symbol"])
def case_go_grouped_var() -> SymbolCase:
    """Go grouped var block."""
    src = """\
package m

var (
	X = 1
	Y = 2
)
"""
    return {"input.go": src}, [("variable", "X", ""), ("variable", "Y", "")]


@case(tags=["symbol"])
def case_go_grouped_const() -> SymbolCase:
    """Go grouped const block (already supported — regression guard)."""
    src = """\
package m

const (
	A = 1
	B = 2
)
"""
    return {"input.go": src}, [("variable", "A", ""), ("variable", "B", "")]


type ReferencesCase = tuple[str, str, list[tuple[str, str, list[str], list[str]]]]


@case(tags=["references"])
def case_go_calls_and_bases() -> ReferencesCase:
    """Each call shape and a generic callee; an embedded field is a base, a named one none."""
    src = """\
package shop

type Order struct {
    Model
    *Base
    name string
}

func (o *Order) Save() {
    helper()
    f.Println()
    exec.Command("x").Run()
    pkg.New[int]()
}
"""
    return (
        "go",
        src,
        [
            ("class", "Order", [], ["unqualified:Model", "unqualified:Base"]),
            (
                "method",
                "Save",
                [
                    "unqualified:helper",
                    "unqualified:f.Println",
                    "unqualified:exec.Command",
                    "unqualified:pkg.New",
                ],
                [],
            ),
        ],
    )

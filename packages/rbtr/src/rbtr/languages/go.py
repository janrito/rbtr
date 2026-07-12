"""Go language plugin.

Provides full support: functions, methods, type declarations,
and import extraction.

Extracted chunks::

    func hello() {}                 → function "hello", scope ""
    func (u User) Name() string {}  → method "Name", scope ""
    type User struct { ... }        → class "User", scope ""
    type Reader interface { ... }   → class "Reader", scope ""

    import "fmt"
        → import, metadata {module: "fmt"}
    import ("fmt" "os/exec")
        → 2 import chunks: {module: "fmt"}, {module: "os/exec"}
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_declaration
  name: (identifier) @_fn_name) @function

(method_declaration
  receiver: (parameter_list
    (parameter_declaration
      type: [
        (type_identifier) @_scope
        (pointer_type (type_identifier) @_scope)
      ]))
  name: (field_identifier) @_method_name) @method

(type_declaration
  (type_spec
    name: (type_identifier) @_cls_name)) @class

(type_spec
  (interface_type
    (method_elem
      name: (field_identifier) @_method_name) @method))

(type_declaration
  (type_alias
    name: (type_identifier) @_cls_name)) @class

(import_declaration
  (import_spec
    path: (interpreted_string_literal) @_import_module) @import)

(import_declaration
  (import_spec_list
    (import_spec
      path: (interpreted_string_literal) @_import_module) @import))

(source_file
  (var_declaration
    (var_spec
      name: (identifier) @_var_name) @variable))

(source_file
  (const_declaration
    (const_spec
      name: (identifier) @_var_name) @variable))

(source_file
  (var_declaration
    (var_spec_list
      (var_spec
        name: (identifier) @_var_name) @variable)))
"""

# ── Plugin ───────────────────────────────────────────────────────────


class GoPlugin:
    """Go language support.

    Uses `type_spec` for scope detection because Go's type
    declarations (`type User struct { ... }`) nest the name
    inside a `type_spec` node within the `type_declaration`.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="go",
                extensions=frozenset({".go"}),
                grammar_module="tree_sitter_go",
                query=_QUERY,
                scope_types=frozenset({"type_spec"}),
                # Go convention: `//` runs directly above a
                # declaration document it (gofmt preserves this
                # link).  The grammar uses a single `comment`
                # type for both line and block forms.
                doc_comment_node_types=frozenset({"comment"}),
                test_suffix="_test",
                language_plugin_version=3,
            ),
        ]

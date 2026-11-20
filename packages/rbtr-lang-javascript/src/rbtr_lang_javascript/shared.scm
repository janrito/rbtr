; Shared patterns — identical across the JS and TS grammars.
; Comments (JS/TS: single `comment` type for `//` and `/* */`, incl. JSDoc)
; at a program's top level and in a class body, where they stand on their
; own or fold into the declaration below. One inside a function or a test's
; callback is that chunk's text.
(program (comment) @comment)
(class_body (comment) @comment)

; A test or a group of tests, at any depth: `describe`, `context`, `it` and
; `test` called with a name, captured on the statement so the call and its
; callback are one chunk, named by the string. A test inside a group is a
; test nested in it.
((expression_statement
  (call_expression
    function: (identifier) @_test_fn
    arguments: (arguments . (string (string_fragment) @_test_name))))
  @test_function
  (#any-of? @_test_fn "describe" "context" "it" "test"))

; The statements a program runs, each one chunk; what is captured inside
; one is a chunk nested in it. A call statement is one when it spans lines
; (a test's statement is also a test, which it is); a one-line call stays
; text around the captures.
(program (if_statement) @statement)
(program (try_statement) @statement)
(program (for_statement) @statement)
(program (for_in_statement) @statement)
(program (while_statement) @statement)
(program
  ((expression_statement (call_expression)) @statement
    (#match? @statement "\n")))

; method_definition covers class and object-literal members, generators,
; and get/set accessors; inside a class it promotes to a method scoped to
; the class, elsewhere it is a scope-less method.
(function_declaration
  name: (identifier) @_fn_name) @function

(generator_function_declaration
  name: (identifier) @_fn_name) @function

; A module's written extension is dropped (`./models.js` is `./models`), so
; it can resolve to a file of another of the language's extensions.
; Every match on one statement is one import. A default import lists and
; binds its local name; a named one lists its name and binds it, under its
; alias when it has one; a namespace import binds the module.
((import_statement
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((import_statement
  (import_clause
    (identifier) @_import_names @_bind_name)
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @_import_names @_bind_name
        !alias)))
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @_import_names @_bind_name
        alias: (identifier) @_bind_local)))
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((import_statement
  (import_clause
    (namespace_import
      (identifier) @_bind_local))
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

(lexical_declaration
  (variable_declarator
    name: (identifier) @_fn_name
    value: (arrow_function))) @function

(method_definition
  name: (property_identifier) @_method_name) @method

; A re-export reads from another module, so it resolves as an import, and
; it is how a symbol reaches a consumer that never names its defining
; file: `export *` offers every name the module does, `export * as ns` the
; module under `ns`, and `export { a, b as bb }` lists `a` and `b`, offered
; as `a` and `bb`.
((export_statement
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((export_statement
  "*"
  source: (string (string_fragment) @_import_module)) @import
  (#set! import.names "*")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((export_statement
  (namespace_export
    (identifier) @_bind_local)
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((export_statement
  (export_clause
    (export_specifier
      name: (identifier) @_import_names @_bind_name
      !alias))
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((export_statement
  (export_clause
    (export_specifier
      name: (identifier) @_import_names @_bind_name
      alias: (identifier) @_bind_local))
  source: (string (string_fragment) @_import_module)) @import
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

; CommonJS: `const x = require("./x")` binds the module under `x`, and
; `const { y, z: zz } = require("./y")` lists `y` and `z`, bound as `y` and
; `zz`; `var` and `let` alike.
((lexical_declaration
  (variable_declarator
    name: (identifier) @_bind_local
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((variable_declaration
  (variable_declarator
    name: (identifier) @_bind_local
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((lexical_declaration
  (variable_declarator
    name: (object_pattern
      (shorthand_property_identifier_pattern) @_import_names @_bind_name)
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((variable_declaration
  (variable_declarator
    name: (object_pattern
      (shorthand_property_identifier_pattern) @_import_names @_bind_name)
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((lexical_declaration
  (variable_declarator
    name: (object_pattern
      (pair_pattern
        key: (property_identifier) @_import_names @_bind_name
        value: (identifier) @_bind_local))
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

((variable_declaration
  (variable_declarator
    name: (object_pattern
      (pair_pattern
        key: (property_identifier) @_import_names @_bind_name
        value: (identifier) @_bind_local))
    value: (call_expression
      function: (identifier) @_require
      arguments: (arguments
        .
        (string (string_fragment) @_import_module))))) @import
  (#eq? @_require "require")
  (#strip! @_import_module "(?<=[^/])\\.[^./]+$"))

; Shared patterns — identical across the JS and TS grammars.
; Comments (JS/TS: single `comment` type for `//` and `/* */`, incl. JSDoc).
(comment) @comment

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

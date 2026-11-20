; Comments and the module docstring. The engine groups comments into
; blank-line-delimited blocks and either folds a block into the symbol flush
; after it (its docstring, possibly nested), drops it if interior, or emits
; it as a standalone COMMENT chunk.
(comment) @comment
(module (expression_statement (string) @comment))

; A decorated definition's chunk is the definition with its decorators, so
; a decorator's call is the definition's; an undecorated one sits directly
; in a module or a block.
(decorated_definition
  definition: (function_definition
    name: (identifier) @_fn_name
    body: (block
      . (expression_statement (string) @_docstring)?))) @function

(module
  (function_definition
    name: (identifier) @_fn_name
    body: (block
      . (expression_statement (string) @_docstring)?)) @function)

(block
  (function_definition
    name: (identifier) @_fn_name
    body: (block
      . (expression_statement (string) @_docstring)?)) @function)

(decorated_definition
  definition: (class_definition
    name: (identifier) @_cls_name
    body: (block
      . (expression_statement (string) @_docstring)?))) @class

(module
  (class_definition
    name: (identifier) @_cls_name
    body: (block
      . (expression_statement (string) @_docstring)?)) @class)

(block
  (class_definition
    name: (identifier) @_cls_name
    body: (block
      . (expression_statement (string) @_docstring)?)) @class)

(type_alias_statement
  . (type (identifier) @_cls_name)) @class

(module
  (expression_statement
    (assignment
      left: (identifier) @_var_name) @variable))

(module
  (expression_statement
    (assignment
      left: (pattern_list (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (tuple_pattern (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (list_pattern (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (pattern_list (list_splat_pattern (identifier) @_var_name))) @variable))

; `import a.b` binds `a`, its first name; `import a.b as c` binds `c` to
; the module.
(import_statement
  name: (dotted_name
    .
    (identifier) @_bind_name) @_import_module) @import

(import_statement
  name: (aliased_import
    name: (dotted_name) @_import_module
    alias: (identifier) @_bind_local)) @import

; `from m import a, b as c` lists `a` and `b` and binds `a` and `c`; a
; relative module is written with its dots, read as parent steps.
(import_from_statement
  module_name: [
    (dotted_name)
    (relative_import)
  ] @_import_module) @import

(import_from_statement
  module_name: [
    (dotted_name)
    (relative_import)
  ] @_import_module
  name: (dotted_name) @_import_names @_bind_name) @import

(import_from_statement
  module_name: [
    (dotted_name)
    (relative_import)
  ] @_import_module
  name: (aliased_import
    name: (dotted_name) @_import_names @_bind_name
    alias: (identifier) @_bind_local)) @import

((import_from_statement
  module_name: [
    (dotted_name)
    (relative_import)
  ] @_import_module
  (wildcard_import)) @import
  (#set! import.names "*"))

; `from __future__ import annotations` has its own node type, and is an
; import like any other.
(future_import_statement
  name: (dotted_name) @_bind_name) @import

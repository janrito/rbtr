; Comments (Rust: `//`/`///`/`//!` line and `/* */` block).
[(line_comment) (block_comment)] @comment

; An attribute documents the item below it and is its sibling in the
; tree, so it folds into that item exactly as a comment block does. One
; standing alone — a module's `#![allow(...)]` — becomes its own chunk.
(source_file [(attribute_item) (inner_attribute_item)] @comment)

(function_item
  name: (identifier) @_fn_name) @function

(function_signature_item
  name: (identifier) @_fn_name) @function

(struct_item
  name: (type_identifier) @_cls_name) @class

(enum_item
  name: (type_identifier) @_cls_name) @class

(enum_item
  body: (enum_variant_list
    (enum_variant
      name: (identifier) @_var_name) @variable))

(union_item
  name: (type_identifier) @_cls_name) @class

(type_item
  name: (type_identifier) @_cls_name) @class

(trait_item
  name: (type_identifier) @_cls_name) @class

; An inline module (`mod net { … }`) is a namespace of definitions.
(mod_item
  name: (identifier) @_cls_name
  body: (declaration_list)) @class

; `mod util;` declares a child of this module, `self::util`, and binds its name.
((mod_item
  !body
  name: (identifier) @_import_module @_bind_local) @import
  (#set! import.prefix "self::"))

(macro_definition
  name: (identifier) @_fn_name) @function

(impl_item
  type: (type_identifier) @_cls_name) @class

; A `use` path is a module written with `::` and the names it lists and
; binds; a leading `super` climbs a level.
; `use a::b::c;` lists and binds `c` from `a::b`; `use serde;` binds the crate.
(use_declaration
  argument: (scoped_identifier
    path: (_) @_import_module
    name: (identifier) @_import_names @_bind_name)) @import

(use_declaration
  argument: (identifier) @_import_module @_bind_name) @import

; `use a::b as c;` binds `b` under `c`; `use a as c;` binds the crate.
(use_declaration
  argument: (use_as_clause
    path: (scoped_identifier
      path: (_) @_import_module
      name: (identifier) @_import_names @_bind_name)
    alias: (identifier) @_bind_local)) @import

(use_declaration
  argument: (use_as_clause
    path: (identifier) @_import_module @_bind_name
    alias: (identifier) @_bind_local)) @import

; `use a::{b, c as d, self};` lists each, `self` binding the module under
; its last name.
(use_declaration
  argument: (scoped_use_list
    path: (_) @_import_module
    list: (use_list
      (identifier) @_import_names @_bind_name))) @import

(use_declaration
  argument: (scoped_use_list
    path: (_) @_import_module
    list: (use_list
      (use_as_clause
        path: (identifier) @_import_names @_bind_name
        alias: (identifier) @_bind_local)))) @import

(use_declaration
  argument: (scoped_use_list
    path: [
      (scoped_identifier
        name: (identifier) @_bind_local)
      (identifier) @_bind_local
    ] @_import_module
    list: (use_list
      (self) @_import_names))) @import

; `use a::*;` lists every public name.
((use_declaration
  argument: (use_wildcard
    [
      (scoped_identifier)
      (identifier)
    ] @_import_module)) @import
  (#set! import.names "*"))

(source_file
  (const_item
    name: (identifier) @_var_name) @variable)

(source_file
  (static_item
    name: (identifier) @_var_name) @variable)

; Top-level comments (Go uses one `comment` type for `//` and `/* */`).
; The engine groups these and folds a block flush before a symbol into it.
(comment) @comment

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

; A declaration of one type is that type, `type` keyword and all; each type
; of a grouped `type ( … )` is a class of its own.
(type_declaration
  "type"
  .
  [
    (type_spec name: (type_identifier) @_cls_name)
    (type_alias name: (type_identifier) @_cls_name)
  ]) @class

(type_declaration
  "("
  [
    (type_spec name: (type_identifier) @_cls_name)
    (type_alias name: (type_identifier) @_cls_name)
  ] @class)

(type_spec
  (interface_type
    (method_elem
      name: (field_identifier) @_method_name) @method))

; An import binds its package under its alias, or, written with none, under
; its path's last segment (`"example.com/m/greeter"` binds `greeter`); `.`
; imports every name and `_` binds nothing.
(import_spec
  name: (package_identifier) @_bind_local
  path: (interpreted_string_literal) @_import_module) @import

((import_spec
  name: (dot)
  path: (interpreted_string_literal) @_import_module) @import
  (#set! import.names "*"))

(import_spec
  name: (blank_identifier)
  path: (interpreted_string_literal) @_import_module) @import

((import_spec
  !name
  path: (interpreted_string_literal) @_import_module @_bind_local) @import
  (#strip! @_bind_local "^\"(?:[^\"]*/)?|\"$"))

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

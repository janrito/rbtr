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

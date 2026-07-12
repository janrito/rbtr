(function_definition
  name: (identifier) @_fn_name
  body: (block
    . (expression_statement (string) @_docstring)?)) @function

(class_definition
  name: (identifier) @_cls_name
  body: (block
    . (expression_statement (string) @_docstring)?)) @class

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

(import_statement
  name: (dotted_name) @_import_module) @import

(import_from_statement
  module_name: (dotted_name) @_import_module) @import

(import_from_statement
  module_name: (relative_import
    (import_prefix) @_import_dots
    (dotted_name) @_import_module)) @import

(import_from_statement
  module_name: (relative_import
    (import_prefix) @_import_dots .)) @import

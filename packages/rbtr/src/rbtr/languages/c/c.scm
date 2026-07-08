
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(preproc_include
  path: (system_lib_string) @_import_module) @import

(preproc_include
  path: (string_literal) @_import_module) @import

(struct_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(union_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(enum_specifier
  name: (type_identifier) @_cls_name
  body: (enumerator_list)) @class

(enumerator
  name: (identifier) @_var_name) @variable

(type_definition
  declarator: (type_identifier) @_cls_name) @class

(type_definition
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (pointer_declarator
        declarator: (type_identifier) @_cls_name)))) @class

(preproc_function_def
  name: (identifier) @_fn_name) @function

(preproc_def
  name: (identifier) @_var_name) @variable

(translation_unit
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_fn_name)) @function)

(translation_unit
  (declaration
    declarator: (init_declarator
      declarator: (identifier) @_var_name)) @variable)

(translation_unit
  (declaration
    declarator: (pointer_declarator
      declarator: (identifier) @_var_name)) @variable)

(translation_unit
  (declaration
    declarator: (init_declarator
      declarator: (pointer_declarator
        declarator: (identifier) @_var_name))) @variable)

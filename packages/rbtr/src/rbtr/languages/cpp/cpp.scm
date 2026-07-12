
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(function_definition
  declarator: (function_declarator
    declarator: (field_identifier) @_fn_name)) @function

(function_definition
  declarator: (function_declarator
    declarator: (operator_name) @_fn_name)) @function

(field_declaration
  declarator: (function_declarator
    declarator: (field_identifier) @_method_name)) @method

(field_declaration_list
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_method_name)) @method)

(preproc_include
  path: (system_lib_string) @_import_module) @import

(preproc_include
  path: (string_literal) @_import_module) @import

(preproc_function_def
  name: (identifier) @_fn_name) @function

(preproc_def
  name: (identifier) @_var_name) @variable

(class_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(struct_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(union_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(enum_specifier
  name: (type_identifier) @_cls_name
  body: (enumerator_list)) @class

(alias_declaration
  name: (type_identifier) @_cls_name) @class

(namespace_definition
  name: (namespace_identifier) @_cls_name) @class

(concept_definition
  name: (identifier) @_cls_name) @class

(translation_unit
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_fn_name)) @function)

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (function_declarator
        declarator: (identifier) @_fn_name)) @function))

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

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (init_declarator
        declarator: (identifier) @_var_name)) @variable))

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (pointer_declarator
        declarator: (identifier) @_var_name)) @variable))

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))) @variable))

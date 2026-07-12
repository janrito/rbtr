(method_declaration
  name: (identifier) @_method_name) @method

(constructor_declaration
  name: (identifier) @_method_name) @method

(field_declaration
  declarator: (variable_declarator
    name: (identifier) @_var_name)) @variable

(class_declaration
  name: (identifier) @_cls_name) @class

(interface_declaration
  name: (identifier) @_cls_name) @class

(enum_declaration
  name: (identifier) @_cls_name) @class

(enum_declaration
  body: (enum_body
    (enum_constant
      name: (identifier) @_var_name) @variable))

(record_declaration
  name: (identifier) @_cls_name) @class

(record_declaration
  parameters: (formal_parameters
    (formal_parameter
      name: (identifier) @_var_name) @variable))

(annotation_type_declaration
  name: (identifier) @_cls_name) @class

(annotation_type_declaration
  body: (annotation_type_body
    (annotation_type_element_declaration
      name: (identifier) @_method_name) @method))

(import_declaration
  (scoped_identifier) @_import_module) @import

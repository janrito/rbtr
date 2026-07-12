; TypeScript adds type-level declarations (interface, enum, type alias,
; abstract class) as classes, and class/interface members as methods.
(class_declaration
  name: (type_identifier) @_cls_name) @class

(abstract_class_declaration
  name: (type_identifier) @_cls_name) @class

(interface_declaration
  name: (type_identifier) @_cls_name) @class

(enum_declaration
  name: (identifier) @_cls_name) @class

(enum_body
  (property_identifier) @_var_name @variable)

(enum_body
  (enum_assignment
    name: (property_identifier) @_var_name) @variable)

(type_alias_declaration
  name: (type_identifier) @_cls_name) @class

(internal_module
  name: (identifier) @_cls_name) @class

(module
  name: (identifier) @_cls_name) @class

(method_signature
  name: (property_identifier) @_method_name) @method

(abstract_method_signature
  name: (property_identifier) @_method_name) @method


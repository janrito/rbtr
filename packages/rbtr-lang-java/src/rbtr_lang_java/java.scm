; Top-level comments (Java: `//` line and `/* */`, `/** */` block).
[(line_comment) (block_comment)] @comment

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

; A class import names its class's file and binds the class.
((import_declaration
  (scoped_identifier
    name: (identifier) @_import_names @_bind_name) @_import_module) @import
  (#not-match? @import "^import\\s+static\\b")
  (#not-match? @import "\\*"))

; A static import names its class's file and binds the member under its
; own name; it lists the member by its path, class then member
; (`util.Strings.shout` lists `Strings.shout`).
((import_declaration
  (scoped_identifier
    scope: (_) @_import_module
    name: (identifier) @_bind_local) @_import_names @_bind_name) @import
  (#match? @import "^import\\s+static\\b")
  (#not-match? @import "\\*")
  (#strip! @_import_names "^.*\\.(?=[^.]+\\.[^.]+$)")
  (#strip! @_bind_name "^.*\\.(?=[^.]+\\.[^.]+$)"))

; `*` imports every class of a package.
((import_declaration
  (scoped_identifier) @_import_module
  (asterisk)) @import
  (#set! import.names "*"))

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

(mod_item
  name: (identifier) @_cls_name) @class

(macro_definition
  name: (identifier) @_fn_name) @function

(impl_item
  type: (type_identifier) @_cls_name) @class

(use_declaration) @import

(source_file
  (const_item
    name: (identifier) @_var_name) @variable)

(source_file
  (static_item
    name: (identifier) @_var_name) @variable)

; Shared patterns — identical across the JS and TS grammars.
; method_definition covers class and object-literal members, generators,
; and get/set accessors; inside a class it promotes to a method scoped to
; the class, elsewhere it is a scope-less method.
(function_declaration
  name: (identifier) @_fn_name) @function

(generator_function_declaration
  name: (identifier) @_fn_name) @function

(import_statement
  source: (string (string_fragment) @_import_module)) @import

(lexical_declaration
  (variable_declarator
    name: (identifier) @_fn_name
    value: (arrow_function))) @function

(method_definition
  name: (property_identifier) @_method_name) @method

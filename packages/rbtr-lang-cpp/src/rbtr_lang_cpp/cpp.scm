; Top-level comments (C++ uses one `comment` type for `//` and `/* */`).
(comment) @comment

; A return type's `*`, `**` or `&` wraps a function's declarator
; (`char* name()` is a `pointer_declarator` around the
; `function_declarator`), so each function pattern also matches it
; wrapped. No deeper wrapping occurs in the C and C++ code measured.
(function_definition
  declarator: [
    (function_declarator
      declarator: [(identifier) (destructor_name)] @_fn_name)
    (pointer_declarator
      declarator: (function_declarator
        declarator: [(identifier) (destructor_name)] @_fn_name))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: [(identifier) (destructor_name)] @_fn_name)))
    (reference_declarator
      (function_declarator
        declarator: [(identifier) (destructor_name)] @_fn_name))
  ]) @function

; A template specialisation (`template <> void encode<int>(…)`) is named by
; its template's name.
(function_definition
  declarator: (function_declarator
    declarator: (template_function
      name: (identifier) @_fn_name))) @function

; A function returning a function pointer (`void (*handler(int))(int)`) is
; named inside the pointer: the outer parameters are the returned pointer's.
(function_definition
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))))) @function

(function_definition
  declarator: [
    (function_declarator
      declarator: (field_identifier) @_fn_name)
    (pointer_declarator
      declarator: (function_declarator
        declarator: (field_identifier) @_fn_name))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: (field_identifier) @_fn_name)))
    (reference_declarator
      (function_declarator
        declarator: (field_identifier) @_fn_name))
  ]) @function

(function_definition
  declarator: [
    (function_declarator
      declarator: (operator_name) @_fn_name)
    (pointer_declarator
      declarator: (function_declarator
        declarator: (operator_name) @_fn_name))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: (operator_name) @_fn_name)))
    (reference_declarator
      (function_declarator
        declarator: (operator_name) @_fn_name))
  ]) @function

; A function whose name is call-shaped, a macro building it
; (`double PREFIX(predict)(…)`) or a C++ call operator read as C
; (`operator()(…)`), is named as written.
(function_definition
  declarator: [
    (function_declarator
      declarator: (function_declarator) @_fn_name)
    (pointer_declarator
      declarator: (function_declarator
        declarator: (function_declarator) @_fn_name))
  ]) @function

; A member defined outside its class (`void Order::save() { … }`), scoped by
; the class it names; `shop::Order::save` is scoped by both, `shop::Order`.
(function_definition
  declarator: [
    (function_declarator
      declarator: (qualified_identifier
        scope: (_) @_scope
        name: [(identifier) (destructor_name) (operator_name)] @_fn_name))
    (pointer_declarator
      declarator: (function_declarator
        declarator: (qualified_identifier
          scope: (_) @_scope
          name: [(identifier) (destructor_name) (operator_name)] @_fn_name)))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: (qualified_identifier
            scope: (_) @_scope
            name: [(identifier) (destructor_name) (operator_name)] @_fn_name))))
    (reference_declarator
      (function_declarator
        declarator: (qualified_identifier
          scope: (_) @_scope
          name: [(identifier) (destructor_name) (operator_name)] @_fn_name)))
  ]) @function

(function_definition
  declarator: [
    (function_declarator
      declarator: (qualified_identifier
        scope: (_) @_scope
        name: (qualified_identifier
          scope: (_) @_scope
          name: [(identifier) (destructor_name) (operator_name)] @_fn_name)))
    (pointer_declarator
      declarator: (function_declarator
        declarator: (qualified_identifier
          scope: (_) @_scope
          name: (qualified_identifier
            scope: (_) @_scope
            name: [(identifier) (destructor_name) (operator_name)] @_fn_name))))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: (qualified_identifier
            scope: (_) @_scope
            name: (qualified_identifier
              scope: (_) @_scope
              name: [(identifier) (destructor_name) (operator_name)] @_fn_name)))))
    (reference_declarator
      (function_declarator
        declarator: (qualified_identifier
          scope: (_) @_scope
          name: (qualified_identifier
            scope: (_) @_scope
            name: [(identifier) (destructor_name) (operator_name)] @_fn_name))))
  ]) @function

(field_declaration
  declarator: [
    (function_declarator
      declarator: [(field_identifier) (operator_name)] @_method_name)
    (pointer_declarator
      declarator: (function_declarator
        declarator: [(field_identifier) (operator_name)] @_method_name))
    (pointer_declarator
      declarator: (pointer_declarator
        declarator: (function_declarator
          declarator: [(field_identifier) (operator_name)] @_method_name)))
    (reference_declarator
      (function_declarator
        declarator: [(field_identifier) (operator_name)] @_method_name))
  ]) @method

(field_declaration_list
  (declaration
    declarator: [
      (function_declarator
        declarator: [(identifier) (destructor_name)] @_method_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: [(identifier) (destructor_name)] @_method_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: [(identifier) (destructor_name)] @_method_name)))
      (reference_declarator
        (function_declarator
          declarator: [(identifier) (destructor_name)] @_method_name))
    ]) @method)

; An included file's every declaration is visible after it.
((preproc_include
  path: [
    (system_lib_string)
    (string_literal)
  ] @_import_module) @import
  (#set! import.names "*"))

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

; A prototype or variable declared at the top of a file, or inside a
; preprocessor conditional (a header's include guard), a namespace or an `extern "C"` block, is
; the file's own; one in a function body is the function's. A query cannot
; give a node pattern alternative parents, so each parent has its patterns.

(translation_unit
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(translation_unit
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(preproc_if
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(preproc_if
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(preproc_ifdef
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(preproc_ifdef
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(preproc_else
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(preproc_else
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(preproc_elif
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(preproc_elif
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(preproc_elifdef
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(preproc_elifdef
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

(declaration_list
  (declaration
    declarator: [
      (function_declarator
        declarator: (identifier) @_fn_name)
      (pointer_declarator
        declarator: (function_declarator
          declarator: (identifier) @_fn_name))
      (pointer_declarator
        declarator: (pointer_declarator
          declarator: (function_declarator
            declarator: (identifier) @_fn_name)))
      (reference_declarator
        (function_declarator
          declarator: (identifier) @_fn_name))
    ]) @function)

(declaration_list
  (declaration
    declarator: [
      (identifier) @_var_name
      (init_declarator
        declarator: (identifier) @_var_name)
      (pointer_declarator
        declarator: (identifier) @_var_name)
      (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))
    ]) @variable)

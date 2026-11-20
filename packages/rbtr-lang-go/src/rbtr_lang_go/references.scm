; What a chunk calls and a struct embeds, for call and inherits edges.
; Mined from tags.scm's `@reference.call`; a callee through a value
; (`exec.Command("x").Run()`) names no path, so the engine drops it.
(call_expression
  function: [
    (identifier)
    (selector_expression)
  ] @_call)

(call_expression
  function: (index_expression
    operand: (_) @_call))

; An embedded field, one with a type and no name, promotes its type's
; methods: the type is a base.
(field_declaration
  !name
  type: [
    (type_identifier) @_base
    (pointer_type (type_identifier) @_base)
    (qualified_type) @_base
  ])

; A function's parameters, its receiver among them, and the names it
; declares are its locals (`@_local`), with their written type as
; `@_local_type`: in `func (o *Order) Save()`, `o.check()` is a call on an
; `Order`. A parameter of another type shape is a local with no type.
(parameter_declaration
  name: (identifier) @_local)

(parameter_declaration
  name: (identifier) @_local
  type: [
    (type_identifier)
    (qualified_type)
  ] @_local_type)

(parameter_declaration
  name: (identifier) @_local
  type: (pointer_type
    [
      (type_identifier)
      (qualified_type)
    ] @_local_type))

(short_var_declaration
  left: (expression_list
    (identifier) @_local))

(short_var_declaration
  left: (expression_list
    .
    (identifier) @_local)
  right: (expression_list
    .
    [
      (composite_literal
        type: [
          (type_identifier)
          (qualified_type)
        ] @_local_type)
      (unary_expression
        operand: (composite_literal
          type: [
            (type_identifier)
            (qualified_type)
          ] @_local_type))
    ]))

(var_spec
  name: (identifier) @_local)

(var_spec
  name: (identifier) @_local
  type: [
    (type_identifier)
    (qualified_type)
  ] @_local_type)

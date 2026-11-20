; What a chunk calls and a class extends, for call and inherits edges.
; Mined from tags.scm's `@reference.call`. A call's receiver and method are
; read as one path; a callee through a value (`items.first.size`) names no
; path, so the engine drops it.
(call
  receiver: (_) @_call
  method: (identifier) @_call @_method
  (#not-eq? @_method "new"))

; `Klass.new` constructs `Klass`: the call reaches the class.
(call
  receiver: [
    (constant)
    (scope_resolution)
  ] @_call
  method: (identifier) @_method
  (#eq? @_method "new"))

(call
  !receiver
  method: (identifier) @_call)

(superclass
  [
    (constant)
    (scope_resolution)
  ] @_base)

; A module a class or module `include`s is a base too, after the
; superclass in written order; Ruby searches them last written first.
((call
  method: (identifier) @_mixin
  arguments: (argument_list
    [
      (constant)
      (scope_resolution)
    ] @_base))
  (#eq? @_mixin "include"))

; `self` is the enclosing class's instance. `super`, bare or with
; arguments, calls the bases' method of the caller's own name.
(self) @_self

(super) @_super_call

; A method's and a block's parameters and the names it assigns are its
; locals (`@_local`); `x = Klass.new` writes `Klass` as the type.
[
  (method_parameters (identifier) @_local)
  (block_parameters (identifier) @_local)
  (lambda_parameters (identifier) @_local)
  (optional_parameter name: (identifier) @_local)
  (keyword_parameter name: (identifier) @_local)
  (splat_parameter name: (identifier) @_local)
  (hash_splat_parameter name: (identifier) @_local)
  (block_parameter name: (identifier) @_local)
  (assignment left: (identifier) @_local)
  (operator_assignment left: (identifier) @_local)
  (assignment left: (left_assignment_list (identifier) @_local))
]

((assignment
  left: (identifier) @_local
  right: (call
    receiver: [
      (constant)
      (scope_resolution)
    ] @_local_type
    method: (identifier) @_new))
  (#eq? @_new "new"))

; A bare name used as a value is a method call unless it is a local, Ruby's
; own rule: `check` alone calls `check`. Only a value's positions are read,
; so a definition's name or a parameter is none.
[
  (body_statement (identifier) @_call)
  (block_body (identifier) @_call)
  (begin_block (identifier) @_call)
  (then (identifier) @_call)
  (else (identifier) @_call)
  (parenthesized_statements (identifier) @_call)
  (argument_list (identifier) @_call)
  (array (identifier) @_call)
  (pair value: (identifier) @_call)
  (interpolation (identifier) @_call)
  (binary left: (identifier) @_call)
  (binary right: (identifier) @_call)
  (unary operand: (identifier) @_call)
  (assignment right: (identifier) @_call)
  (operator_assignment right: (identifier) @_call)
  (if_modifier condition: (identifier) @_call)
  (unless_modifier condition: (identifier) @_call)
  (while_modifier condition: (identifier) @_call)
  (until_modifier condition: (identifier) @_call)
  (if condition: (identifier) @_call)
  (unless condition: (identifier) @_call)
  (while condition: (identifier) @_call)
  (until condition: (identifier) @_call)
  (conditional condition: (identifier) @_call)
  (conditional consequence: (identifier) @_call)
  (conditional alternative: (identifier) @_call)
  (element_reference object: (identifier) @_call)
]

; What a chunk calls and what a class extends, for call and inherits edges.
; Mined from tags.scm's `@reference.call`, which takes a name or an attribute
; callee; any callee is captured here, and one through a value (`x[0].m()`)
; names no path, so the engine drops it.
(call
  function: (_) @_call)

; A base is a name or an attribute; `metaclass=Meta` is a keyword argument.
(class_definition
  superclasses: (argument_list
    [
      (identifier)
      (attribute)
    ] @_base))

; `self` and `cls` name the enclosing class's instance and class, and
; `super` reaches its bases: a path starting at one starts from them.
((identifier) @_self
  (#any-of? @_self "self" "cls"))

((identifier) @_super
  (#eq? @_super "super"))

; A function's parameters and the names it assigns are its locals
; (`@_local`), with their written type, an annotation or the callee assigned
; to them, as `@_local_type`: a call through one is on a value.
(parameters
  [
    (identifier) @_local
    (default_parameter name: (identifier) @_local)
    (list_splat_pattern (identifier) @_local)
    (dictionary_splat_pattern (identifier) @_local)
  ])

(parameters
  [
    (typed_parameter
      (identifier) @_local
      type: (type) @_local_type)
    (typed_default_parameter
      name: (identifier) @_local
      type: (type) @_local_type)
  ])

(assignment
  left: (identifier) @_local)

(assignment
  left: (identifier) @_local
  type: (type) @_local_type)

(assignment
  left: (identifier) @_local
  right: (call
    function: [
      (identifier)
      (attribute)
    ] @_local_type))

(assignment
  left: [
    (pattern_list
      (identifier) @_local)
    (tuple_pattern
      (identifier) @_local)
  ])

(for_statement
  left: [
    (pattern_list
      (identifier) @_local)
    (tuple_pattern
      (identifier) @_local)
  ])

(augmented_assignment
  left: (identifier) @_local)

(for_statement
  left: (identifier) @_local)

(for_in_clause
  left: (identifier) @_local)

(as_pattern
  alias: (as_pattern_target
    (identifier) @_local))

; What a chunk calls and a class extends, for call and inherits edges.
; Mined from tags.scm's `@reference.call` and `@reference.class`. Java's
; `method_invocation` has no single callee node, so the object and the name
; are captured together and read as one path; a callee through a value
; (`items.get(0).size()`) names no path, so the engine drops it.
(method_invocation
  object: (_) @_call
  name: (identifier) @_call)

(method_invocation
  !object
  name: (identifier) @_call)

(object_creation_expression
  type: [
    (type_identifier) @_call
    (generic_type (type_identifier) @_call)
  ])

; A class's bases are its superclass, then the interfaces it implements; an
; interface's are the interfaces it extends. Written in that order, they are
; searched in it: a class's own chain before an interface's default method.
(superclass
  [
    (type_identifier) @_base
    (generic_type (type_identifier) @_base)
    (scoped_type_identifier) @_base
  ])

(super_interfaces
  (type_list
    [
      (type_identifier) @_base
      (generic_type (type_identifier) @_base)
      (scoped_type_identifier) @_base
    ]))

(extends_interfaces
  (type_list
    [
      (type_identifier) @_base
      (generic_type (type_identifier) @_base)
      (scoped_type_identifier) @_base
    ]))

; `this` is the enclosing class's instance and `super` its base.
(this) @_self

(super) @_super

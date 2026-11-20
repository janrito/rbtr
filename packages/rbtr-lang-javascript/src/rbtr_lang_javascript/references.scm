; What a chunk calls, for call edges: a call's callee, and the class a
; `new` constructs. Mined from tags.scm's `@reference.call` and
; `@reference.class`; a callee through a value (`x[0].m()`) names no path,
; so the engine drops it.
(call_expression
  function: (_) @_call)

(new_expression
  constructor: (_) @_call)

; `this` is the enclosing class's instance and `super` its base: a path
; starting at one starts from them.
(this) @_self

(super) @_super

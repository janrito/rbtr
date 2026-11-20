; What a chunk calls, for call edges. C has no classes: a call through a
; struct's field (`s.m()`, `p->m()`) starts at a value, so its head reaches
; nothing.
(call_expression
  function: [
    (identifier)
    (field_expression)
  ] @_call)

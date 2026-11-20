; What a chunk calls and a class extends, for call and inherits edges.
; Mined from tags.scm's `@reference.call`; a template call (`make<int>()`)
; by its name, and the class a `new` constructs.
(call_expression
  function: [
    (identifier)
    (field_expression)
    (qualified_identifier)
  ] @_call)

(call_expression
  function: (template_function
    name: (_) @_call))

(new_expression
  type: [
    (type_identifier)
    (qualified_identifier)
  ] @_call)

; A class's bases, after their access specifiers.
(base_class_clause
  [
    (type_identifier)
    (qualified_identifier)
  ] @_base)

; `this` is the enclosing class's instance.
(this) @_self

; What a chunk calls, for call edges. Mined from tags.scm's
; `@reference.call`; a generic call (`foo::<u8>()`) by its function, and a
; callee through a value (`x.y().z()`) names no path, so the engine drops
; it. A macro invocation (`println!`) is no call. Rust has no inheritance:
; a trait an impl implements is no base.
(call_expression
  function: [
    (identifier)
    (scoped_identifier)
    (field_expression)
  ] @_call)

(call_expression
  function: (generic_function
    function: (_) @_call))

; `self` is the impl's instance and `Self` its type; Rust's `super` is the
; parent module, not a base, so nothing marks it.
(self) @_self

((identifier) @_self
  (#eq? @_self "Self"))

; `impl Trait for Type` gives the type the trait's methods: the trait is a
; base of the impl block, which is a class named by the type. A trait's
; supertraits (`trait Titled: Named`) are its bases.
(impl_item
  trait: [
    (type_identifier) @_base
    (scoped_type_identifier) @_base
    (generic_type type: [
      (type_identifier)
      (scoped_type_identifier)
    ] @_base)
  ])

(trait_item
  bounds: (trait_bounds
    [
      (type_identifier)
      (scoped_type_identifier)
    ] @_base))

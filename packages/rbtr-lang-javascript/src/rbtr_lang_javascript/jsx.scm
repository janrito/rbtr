; A component a JSX element renders is a reference, as a call: `<Toolbar>`,
; `<Ui.Button />`. A lowercase name (`<div>`) is an HTML element, not a
; component; a dotted one (`<motion.div>`) always names a component.
((jsx_opening_element
  name: (identifier) @_call)
  (#match? @_call "^[A-Z]"))

((jsx_self_closing_element
  name: (identifier) @_call)
  (#match? @_call "^[A-Z]"))

(jsx_opening_element
  name: (member_expression) @_call)

(jsx_self_closing_element
  name: (member_expression) @_call)

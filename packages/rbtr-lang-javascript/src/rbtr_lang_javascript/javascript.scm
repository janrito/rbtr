; JS uses `identifier` for class names (TS uses `type_identifier`).
(class_declaration
  name: (identifier) @_cls_name) @class


; A class's base, for inherits edges: `class A extends B`.
(class_heritage
  [
    (identifier)
    (member_expression)
  ] @_base)

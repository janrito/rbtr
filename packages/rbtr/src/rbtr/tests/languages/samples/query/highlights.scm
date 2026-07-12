; A tree-sitter query file exercising every construct rbtr should index:
; named nodes with fields, anonymous nodes, alternation lists, groupings,
; predicates, wildcards, negated fields, anchors, quantifiers, supertypes,
; and a MISSING-node pattern. Every top-level pattern is one doc section.

; Named-node patterns with a field and an outer definition capture.
(function_definition
  name: (identifier) @function
  body: (_) @_body) @definition.function

(class_definition
  name: (identifier) @type) @definition.class

; Alternation list and a bare anonymous-node matcher.
["if" "else" "while" "return"] @keyword

"async" @keyword.coroutine

; Wildcard child, a negated field, anchors, and quantifiers.
(call_expression
  function: (_) @call
  !type_arguments)

(array . (identifier) @first (_)* @rest)

; Groupings with predicates: regex match, alternation, and an injection.
((identifier) @constant
  (#match? @constant "^[A-Z][A-Z_]+$"))

((identifier) @type.builtin
  (#any-of? @type.builtin "int" "str" "bool"))

((comment) @injection.content
  (#set! injection.language "markdown"))

; Supertype/subtype match and a MISSING-node pattern.
(declaration/function_definition) @definition.function

(binary_expression (MISSING identifier))

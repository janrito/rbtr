; Every top-level pattern is a section. A pattern is named by its own outer
; label — a capture that is a direct child of the pattern node (e.g.
; `(function_definition …) @function` → "function"; `["if" "else"] @keyword`
; → "keyword"). The `?` makes that label optional: predicate-wrapped rules
; and structural wrappers carry no label of their own and stay anonymous.
; Captures, predicates, and matched node types inside remain full-text
; searchable within the section either way.
(program (named_node (capture (identifier) @_section_name)?) @doc_section)
(program (list (capture (identifier) @_section_name)?) @doc_section)
(program (grouping (capture (identifier) @_section_name)?) @doc_section)
(program (anonymous_node (capture (identifier) @_section_name)?) @doc_section)

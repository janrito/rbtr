; Module-level const/let bindings with a non-function value become
; variables. The value allowlist excludes arrow/function expressions,
; which shared.scm already captures as functions. Flat destructuring
; targets are captured; nested patterns are not (no query-only recursion).

(program
  (lexical_declaration
    (variable_declarator
      name: (identifier) @_var_name
      value: [(number) (string) (template_string) (true) (false) (null)
 (object) (array) (identifier) (member_expression)
 (call_expression) (new_expression) (binary_expression) (unary_expression)]) @variable))

(program
  (export_statement
    declaration: (lexical_declaration
      (variable_declarator
        name: (identifier) @_var_name
        value: [(number) (string) (template_string) (true) (false) (null)
 (object) (array) (identifier) (member_expression)
 (call_expression) (new_expression) (binary_expression) (unary_expression)]) @variable)))

(program
  (lexical_declaration
    (variable_declarator
      name: [
  (object_pattern [
    (shorthand_property_identifier_pattern) @_var_name
    (pair_pattern value: (identifier) @_var_name)
    (object_assignment_pattern left: (shorthand_property_identifier_pattern) @_var_name)
    (rest_pattern (identifier) @_var_name)
  ])
  (array_pattern [
    (identifier) @_var_name
    (rest_pattern (identifier) @_var_name)
  ])
]) @variable))

(program
  (export_statement
    declaration: (lexical_declaration
      (variable_declarator
        name: [
  (object_pattern [
    (shorthand_property_identifier_pattern) @_var_name
    (pair_pattern value: (identifier) @_var_name)
    (object_assignment_pattern left: (shorthand_property_identifier_pattern) @_var_name)
    (rest_pattern (identifier) @_var_name)
  ])
  (array_pattern [
    (identifier) @_var_name
    (rest_pattern (identifier) @_var_name)
  ])
]) @variable)))

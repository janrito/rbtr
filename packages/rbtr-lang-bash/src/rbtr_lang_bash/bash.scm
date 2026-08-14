; Top-level comments (Bash: single `comment` type).
(comment) @comment

(function_definition
  name: (word) @_fn_name) @function

(command
  name: (command_name
    (word) @_cmd)
  .
  (word) @_import_module
  (#eq? @_cmd "source")) @import

(command
  name: (command_name
    (word) @_cmd)
  .
  (word) @_import_module
  (#eq? @_cmd ".")) @import

(program
  (variable_assignment
    name: (variable_name) @_var_name) @variable)

(program
  (declaration_command
    (variable_assignment
      name: (variable_name) @_var_name)) @variable)

(command
  name: (command_name (word) @_cmd)
  argument: (concatenation (word) @_var_name)
  (#eq? @_cmd "alias")
  (#match? @_var_name "=")) @variable

; A script's top-level statements are what it does, and no name stands
; for them, so each is an anonymous section located by its lines. A
; script that defines nothing is still entirely searchable.
(program
  (command
    name: (command_name (word) @_statement)
    ; `source`, `.` and `alias` are already an import or a variable above.
    ; One `#not-match?` rather than several `#not-any-of?` arguments, which
    ; this binding applies only when given exactly one.
    (#not-match? @_statement "^(source|[.]|alias)$")) @doc_section)

(program
  [(if_statement) (for_statement) (while_statement) (case_statement)
   (pipeline) (list) (subshell) (compound_statement) (unset_command)
   (redirected_statement)] @doc_section)

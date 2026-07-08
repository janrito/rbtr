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

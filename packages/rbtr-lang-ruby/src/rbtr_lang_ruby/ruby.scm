
; Comments (Ruby: `#` line and `=begin`/`=end` block, both `comment`).
(comment) @comment

(method
  name: (identifier) @_fn_name) @function

(singleton_method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class

(module
  name: (constant) @_cls_name) @class

; A required file's constants are visible to every file after it: its
; every name, as a star import. `require_relative` reads its path from the
; requiring file's directory.
((call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)) @import
  (#set! import.names "*")
  (#eq? @_call_name "require")
  (#not-match? @_import_module "^.{0,2}$"))

((call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)) @import
  (#set! import.names "*")
  (#set! import.prefix "./")
  (#eq? @_call_name "require_relative")
  (#not-match? @_import_module "^.{0,2}$"))

; An empty path names nothing.
((call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)) @import
  (#any-of? @_call_name "require" "require_relative")
  (#match? @_import_module "^.{0,2}$"))

(assignment
  left: (constant) @_var_name) @variable

(assignment
  left: (left_assignment_list (constant) @_var_name)) @variable

(assignment
  left: (left_assignment_list (rest_assignment (constant) @_var_name))) @variable

(call
  method: (identifier) @_call_name
  arguments: (argument_list . (string (string_content) @_cls_name))
  (#any-of? @_call_name "describe" "context" "feature" "shared_examples" "shared_context")) @class

(call
  method: (identifier) @_call_name
  arguments: (argument_list . (constant) @_cls_name)
  (#any-of? @_call_name "describe" "context" "feature")) @class

(call
  method: (identifier) @_call_name
  arguments: (argument_list . (string (string_content) @_fn_name))
  (#any-of? @_call_name "it" "specify" "example" "scenario")) @function

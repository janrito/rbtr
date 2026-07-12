
(method
  name: (identifier) @_fn_name) @function

(singleton_method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class

(module
  name: (constant) @_cls_name) @class

(call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)
  (#eq? @_call_name "require"))  @import

(call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)
  (#eq? @_call_name "require_relative"))  @import

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

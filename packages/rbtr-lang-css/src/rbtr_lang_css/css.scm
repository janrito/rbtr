; Comments (`/* */`).
(comment) @comment

(rule_set
  (selectors) @_cls_name) @class

(media_statement) @class

(charset_statement) @config_key

(keyframes_statement
  (keyframes_name) @_cls_name) @class

(import_statement
  (call_expression
    (arguments
      (string_value (string_content) @_import_module)))) @import

(import_statement
  (string_value (string_content) @_import_module)) @import

(declaration
  (property_name) @_var_name
  (#match? @_var_name "^--")) @variable

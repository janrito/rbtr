
; Comments (Less: `/* */` is `comment`, `//` is `js_comment`).
[(comment) (js_comment)] @comment

(declaration
  (property_name) @_var_name
  (#match? @_var_name "^[@]")) @variable

(mixin_definition
  (class_name) @_fn_name) @function

(rule_set
  (selectors) @_cls_name) @class

(media_statement) @class

(charset_statement) @config_key

(keyframes_statement
  (keyframes_name) @_cls_name) @class

(import_statement
  (string_value) @_import_module) @import

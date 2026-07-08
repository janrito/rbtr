
(declaration
  (property_name) @_var_name
  (#match? @_var_name "^[@]")) @variable

(mixin_definition
  (class_name) @_fn_name) @function

(rule_set
  (selectors) @_section_name) @doc_section

(media_statement) @doc_section

(charset_statement) @doc_section

(keyframes_statement
  (keyframes_name) @_section_name) @doc_section

(import_statement
  (string_value) @_import_module) @import

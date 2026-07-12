(rule_set
  (selectors) @_section_name) @doc_section

(media_statement) @doc_section

(charset_statement) @doc_section

(keyframes_statement
  (keyframes_name) @_section_name) @doc_section

(import_statement
  (call_expression
    (arguments
      (string_value (string_content) @_import_module)))) @import

(import_statement
  (string_value (string_content) @_import_module)) @import

(declaration
  (property_name) @_var_name
  (#match? @_var_name "^--")) @variable

(pair
  key: (string (string_content) @_section_name)) @config_key

; A `$ref` names another schema document, and `extends` names a base
; configuration: both are references to a file this repository may hold.
(pair
  key: (string (string_content) @_key)
  (#any-of? @_key "$ref" "extends")
  value: (string (string_content) @_import_module)) @import

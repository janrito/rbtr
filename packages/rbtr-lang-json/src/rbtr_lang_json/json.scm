(pair
  key: (string (string_content) @_section_name)) @config_key

; A `$ref` names another schema document, and `extends` names a base
; configuration: both are references to a file this repository may hold.
; After a `#` comes a JSON pointer, which names its last key: keys are not
; scoped by their parents (`defs.json#/definitions/Named` names `Named`).
((pair
  key: (string (string_content) @_key)
  value: (string (string_content) @_import_module @_import_names)) @import
  (#strip! @_import_names "^[^#]*#?(?:.*/)?")
  (#any-of? @_key "$ref" "extends"))

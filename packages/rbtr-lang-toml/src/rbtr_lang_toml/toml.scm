; Comments (`#`).
(comment) @comment

(table
  (bare_key) @_section_name) @config_key

(table
  (dotted_key) @_section_name) @config_key

(table
  (quoted_key) @_section_name) @config_key

(table_array_element
  (bare_key) @_section_name) @config_key

(table_array_element
  (dotted_key) @_section_name) @config_key

(table_array_element
  (quoted_key) @_section_name) @config_key

; A `path` dependency names a directory this repository holds, as Cargo
; writes a local crate: `helper = { path = "crates/helper" }`.
(pair
  (bare_key) @_key
  (#eq? @_key "path")
  (string) @_import_module) @import

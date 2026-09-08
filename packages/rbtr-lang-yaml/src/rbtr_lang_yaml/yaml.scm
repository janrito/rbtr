; Comments (`#`).
(comment) @comment

(document
  (block_node
    (block_mapping
      (block_mapping_pair
        key: (_) @_section_name) @config_key)))

; A `$ref` names another document this repository may hold.
(block_mapping_pair
  key: (flow_node) @_key
  (#eq? @_key "$ref")
  value: (flow_node) @_import_module) @import

; A `uses` naming a path reads a workflow or action from this repository;
; one naming anything else reads it from elsewhere, as an external link in
; prose does, and is left alone.
(block_mapping_pair
  key: (flow_node) @_key
  (#eq? @_key "uses")
  value: (flow_node) @_import_module
  (#match? @_import_module "^\\.{1,2}/")) @import

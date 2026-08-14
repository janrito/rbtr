; Comments (HCL: `#`, `//`, `/* */` — all `comment`).
(comment) @comment

(config_file
  (body
    (block) @config_key))

; A `module` block's `source` names the directory the module is read from,
; so it is an import: `module "x" { source = "./b" }` reaches `./b`.
(block
  (identifier) @_block_type
  (#eq? @_block_type "module")
  (body
    (attribute
      (identifier) @_attr
      (#eq? @_attr "source")
      (expression
        (literal_value
          (string_lit
            (template_literal) @_import_module)))))) @import

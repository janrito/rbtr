; <script>/<style> blocks delegate to JS/TS/CSS/SCSS/Less. A `lang="…"`
; attribute selects the dialect (priority 1); the bare block is the
; fallback (priority 0).
;
; `#set!` comes before a pattern's predicates: py-tree-sitter 0.26.1 frees
; the predicate before a `#set!` twice, and a later query crashes (the
; `rbtr-languages` skill, "Gotchas").

((script_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#set! injection.language "typescript")
 (#set! injection.priority "1")
 (#eq? @_attr "lang")
 (#any-of? @_lang "ts" "typescript"))

((script_element (raw_text) @injection.content)
 (#set! injection.language "javascript")
 (#set! injection.priority "0"))

((style_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#set! injection.language "scss")
 (#set! injection.priority "1")
 (#eq? @_attr "lang")
 (#any-of? @_lang "scss" "sass"))

((style_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#set! injection.language "less")
 (#set! injection.priority "1")
 (#eq? @_attr "lang")
 (#any-of? @_lang "less"))

((style_element (raw_text) @injection.content)
 (#set! injection.language "css")
 (#set! injection.priority "0"))

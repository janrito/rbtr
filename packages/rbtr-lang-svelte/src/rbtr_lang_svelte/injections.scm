; <script>/<style> blocks delegate to JS/TS/CSS/SCSS/Less. A `lang="…"`
; attribute selects the dialect (priority 1); the bare block is the
; fallback (priority 0).

((script_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#eq? @_attr "lang")
 (#any-of? @_lang "ts" "typescript")
 (#set! injection.language "typescript")
 (#set! injection.priority "1"))

((script_element (raw_text) @injection.content)
 (#set! injection.language "javascript")
 (#set! injection.priority "0"))

((style_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#eq? @_attr "lang")
 (#any-of? @_lang "scss" "sass")
 (#set! injection.language "scss")
 (#set! injection.priority "1"))

((style_element
   (start_tag
     (attribute (attribute_name) @_attr
       (quoted_attribute_value (attribute_value) @_lang)))
   (raw_text) @injection.content)
 (#eq? @_attr "lang")
 (#any-of? @_lang "less")
 (#set! injection.language "less")
 (#set! injection.priority "1"))

((style_element (raw_text) @injection.content)
 (#set! injection.language "css")
 (#set! injection.priority "0"))

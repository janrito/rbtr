; Comments (`<!-- -->`).
(comment) @comment

(element (start_tag (tag_name) @_tag)
  (#any-of? @_tag "head" "body" "article" "aside" "nav" "section" "header" "main" "footer" "figure" "form" "dialog" "details" "table")) @doc_section

(script_element
  (start_tag
    (attribute (attribute_name) @_a
      (quoted_attribute_value (attribute_value) @_import_module))
    (#eq? @_a "src"))) @import

(element
  [(start_tag (tag_name) @_lt
     (attribute (attribute_name) @_h
       (quoted_attribute_value (attribute_value) @_import_module)))
   (self_closing_tag (tag_name) @_lt
     (attribute (attribute_name) @_h
       (quoted_attribute_value (attribute_value) @_import_module)))]
  (#eq? @_lt "link")
  (#eq? @_h "href")) @import

; A link to another page in the repository. An external URL (a scheme or
; `//`) is not a cross-reference, and a link to an anchor on its own page
; (`#top`) names no file yet.
(element
  (start_tag (tag_name) @_at
    (attribute (attribute_name) @_ah
      (quoted_attribute_value (attribute_value) @_import_module)))
  (#eq? @_at "a")
  (#eq? @_ah "href")
  (#not-match? @_import_module "^([a-zA-Z][a-zA-Z0-9+.-]*:|#|//)")) @import

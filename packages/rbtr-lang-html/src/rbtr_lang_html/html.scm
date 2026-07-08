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

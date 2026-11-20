; Comments (`<!-- -->`).
(comment) @comment

(element (start_tag (tag_name) @_tag)
  (#any-of? @_tag "head" "body" "article" "aside" "nav" "section" "header" "main" "footer" "figure" "form" "dialog" "details" "table")) @doc_section

; Any other element with an `id` or `name`, which a link's fragment names.
; `#not-match?`, not `#not-any-of?`: py-tree-sitter 0.26.1 excludes nothing
; with a `#not-any-of?` of several values.
(element
  (start_tag (tag_name) @_tag
    (attribute (attribute_name) @_a
      (quoted_attribute_value (attribute_value) @_section_name)))
  (#any-of? @_a "id" "name")
  (#not-match? @_tag "^(head|body|article|aside|nav|section|header|main|footer|figure|form|dialog|details|table)$")) @doc_section

; A start tag alone, as inline HTML in Markdown gives one (`<a id="x">`),
; parses as an error holding the tag.
(ERROR
  (start_tag
    (attribute (attribute_name) @_a
      (quoted_attribute_value (attribute_value) @_section_name)))
  (#any-of? @_a "id" "name")) @doc_section

; A script's `src` imports JavaScript, a stylesheet link's `href` CSS: each
; is hinted so it resolves by that language's rules.
((script_element
  (start_tag
    (attribute (attribute_name) @_a
      (quoted_attribute_value (attribute_value) @_import_module)))) @import
  (#set! import.language_hint "javascript")
  (#eq? @_a "src"))

((element
  [(start_tag (tag_name) @_lt
     (attribute (attribute_name) @_h
       (quoted_attribute_value (attribute_value) @_import_module)))
   (self_closing_tag (tag_name) @_lt
     (attribute (attribute_name) @_h
       (quoted_attribute_value (attribute_value) @_import_module)))]) @import
  (#set! import.language_hint "css")
  (#eq? @_lt "link")
  (#eq? @_h "href"))

; A link to a page in the repository, or to an element on its own page
; (`#top`). An external URL (a scheme or `//`) is not a cross-reference.
(element
  (start_tag (tag_name) @_at
    (attribute (attribute_name) @_ah
      (quoted_attribute_value (attribute_value) @_import_module)))
  (#eq? @_at "a")
  (#eq? @_ah "href")
  (#not-match? @_import_module "^([a-zA-Z][a-zA-Z0-9+.-]*:|//)")) @import

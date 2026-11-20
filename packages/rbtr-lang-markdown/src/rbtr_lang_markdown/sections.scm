; A heading in a document's sections is a flat section at its level: `#` to
; `######` are 1 to 6, an `=` underline 1 and a `-` underline 2. The grammar
; nests sections for `#` headings only, and gives an underlined heading no
; section of its own, so each heading is levelled as written: it holds its
; text up to the next heading and sits under the nearest earlier one of a
; higher level, whichever style either uses.
([
  (document (section . (atx_heading (atx_h1_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h1_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "1"))

([
  (document (section . (atx_heading (atx_h2_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h2_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "2"))

([
  (document (section . (atx_heading (atx_h3_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h3_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "3"))

([
  (document (section . (atx_heading (atx_h4_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h4_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "4"))

([
  (document (section . (atx_heading (atx_h5_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h5_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "5"))

([
  (document (section . (atx_heading (atx_h6_marker) @_level (inline) @_section_name) @doc_section))
  (section (section . (atx_heading (atx_h6_marker) @_level (inline) @_section_name) @doc_section))
]
  (#set! level "6"))

([
  (document (section (setext_heading heading_content: (paragraph (inline) @_section_name) (setext_h1_underline) @_level) @doc_section))
  (section (section (setext_heading heading_content: (paragraph (inline) @_section_name) (setext_h1_underline) @_level) @doc_section))
]
  (#set! level "1"))

([
  (document (section (setext_heading heading_content: (paragraph (inline) @_section_name) (setext_h2_underline) @_level) @doc_section))
  (section (section (setext_heading heading_content: (paragraph (inline) @_section_name) (setext_h2_underline) @_level) @doc_section))
]
  (#set! level "2"))

; A heading inside a list item or a block quote is a section within the one
; around it, nested by the grammar.
([
  (list_item (section (atx_heading (inline) @_section_name)) @doc_section)
  (block_quote (section (atx_heading (inline) @_section_name)) @doc_section)
])

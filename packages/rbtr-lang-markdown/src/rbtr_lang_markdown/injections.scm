; Fenced code blocks: the info string names the embedded language.
(fenced_code_block
  (info_string (language) @injection.language)
  (code_fence_content) @injection.content)

; A block's inline text is parsed by the grammar's inline half, which
; extracts its links.
((inline) @injection.content
  (#set! injection.language "markdown_inline"))

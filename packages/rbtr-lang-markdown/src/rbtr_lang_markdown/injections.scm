; Fenced code blocks: the info string names the embedded language.
(fenced_code_block
  (info_string (language) @injection.language)
  (code_fence_content) @injection.content)

; Raw HTML on lines of its own is HTML.
((html_block) @injection.content
  (#set! injection.language "html"))

; A block's inline text is parsed by the grammar's inline half, whose own
; injections reach the HTML inside it (`inline_injections.scm`).
((inline) @injection.content
  (#set! injection.language "markdown_inline"))

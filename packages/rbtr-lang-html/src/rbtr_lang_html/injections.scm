; Inline <script>/<style> delegate to JavaScript/CSS. External
; <script src>/<link href> have no raw_text, so they become imports instead.

((script_element (raw_text) @injection.content)
  (#set! injection.language "javascript"))

((style_element (raw_text) @injection.content)
  (#set! injection.language "css"))

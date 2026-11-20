; A link to a file, or to a section of one (`guide.md#setup`, `#usage`). The
; destination is the import. An external URL (a scheme, `//`) is not a
; cross-reference.
(inline_link
  (link_destination) @_import_module @import
  (#not-match? @_import_module "^([a-zA-Z][a-zA-Z0-9+.-]*:|//)"))

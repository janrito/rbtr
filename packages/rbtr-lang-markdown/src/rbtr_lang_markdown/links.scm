; A link to a file, or to a section of one (`guide.md#setup`). The
; destination is the import. An external URL (a scheme, `//`), and a link
; to a section of its own page (`#usage`), are not recorded.
(inline_link
  (link_destination) @_import_module @import
  (#not-match? @_import_module "^([a-zA-Z][a-zA-Z0-9+.-]*:|//|#)"))

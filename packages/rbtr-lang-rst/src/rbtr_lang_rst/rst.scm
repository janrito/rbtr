; A section: its title, and its adornments as `@_level`. tree-sitter-rst
; leaves sections flat (a `section` holds its title alone), so the engine
; levels them by the order each style first appears: the adornment's
; character, twice when the title is overlined.
((section "adornment"? @_level (title) @_section_name "adornment" @_level
  (#strip! @_level "(?<=^.).*")) @doc_section)

; A role naming a symbol: `:func:`format_greeting``, `:class:`~pkg.Greeter``
; (`~` shows the last component, which is the name).
((interpreted_text (role) @_role "interpreted_text" @_import_names) @import
  (#any-of? @_role ":func:" ":class:" ":meth:" ":attr:")
  (#strip! @_import_names "^`~[^`]*\\.|^`|`$"))

; A role naming a document or module: `:doc:`topics/intro``.
((interpreted_text (role) @_role "interpreted_text" @_import_module) @import
  (#any-of? @_role ":doc:" ":mod:")
  (#strip! @_import_module "^`|`$"))

; A hyperlink with an embedded target in the repository:
; `the guide <guide.rst#setup>`_. An external URL (a scheme, `//`) is not a
; cross-reference.
((reference) @_import_module @import
  (#match? @_import_module "<[^>]*>`_+$")
  (#not-match? @_import_module "<([a-zA-Z][a-zA-Z0-9+.-]*:|//)")
  (#strip! @_import_module "^`[^<]*<|>`_+$"))

; A `toctree`'s entries, one per line (`intro`, `Reference <api/index>`).
; tree-sitter-rst tokenises directive content by word, so the entries are
; the quantified `text` nodes, which the engine reads one module per line.
((directive (type) @_type
  (body (content "text"+ @_import_module)) @import)
  (#eq? @_type "toctree")
  (#strip! @_import_module "^[^<]*<|>$"))

(comment) @comment

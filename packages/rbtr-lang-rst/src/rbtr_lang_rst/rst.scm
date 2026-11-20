; A section: its title, and its adornments as `@_level`. tree-sitter-rst
; leaves sections flat (a `section` holds its title alone), so the engine
; levels them by the order each style first appears: the adornment's
; character, twice when the title is overlined.
((section "adornment"? @_level (title) @_section_name "adornment" @_level
  (#strip! @_level "(?<=^.).*")) @doc_section)

; A label or hyperlink target (`.. _intro:`, `.. _docs: https://…`): a
; section named as a `:ref:` role or a link names it.
((target (name) @_section_name) @doc_section
  (#strip! @_section_name "^_|:$"))

; A role naming a Python object: `:func:`format_greeting``,
; `:class:`~pkg.Greeter`` (`~` shows the last component, which is the name).
; The name lives in Python, Sphinx's default domain.
; (`#set!` first: py-tree-sitter 0.26.1 frees a predicate before it twice.)
((interpreted_text (role) @_role "interpreted_text" @_import_names) @import
  (#set! import.language_hint "python")
  (#any-of? @_role ":func:" ":class:" ":meth:" ":attr:")
  (#strip! @_import_names "^`~[^`]*\\.|^`|`$"))

; A role naming a document or module: `:doc:`topics/intro``.
((interpreted_text (role) @_role "interpreted_text" @_import_module) @import
  (#any-of? @_role ":doc:" ":mod:")
  (#strip! @_import_module "^`|`$"))

; A role naming a label, wherever in the documents it is set:
; `:ref:`intro``, `:ref:`the intro <intro>``. The name lives in rST.
((interpreted_text (role) @_role "interpreted_text" @_import_names) @import
  (#set! import.language_hint "rst")
  (#eq? @_role ":ref:")
  (#strip! @_import_names "^`([^`<]*<)?|>?`$"))

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

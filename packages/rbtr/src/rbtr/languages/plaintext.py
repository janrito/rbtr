"""The language of a file no plugin claims.

It registers no extensions and no filenames, so detection never routes
to it by name: the build resolves to it once extension, stored detection
and the prose sniff have all found nothing. It has no grammar and no
query, so extraction falls to `chunk_plaintext`, whose line slices carry
`PLAINTEXT` as their language.

Being a registration like any other, it is discovered through the
`rbtr.languages` entry point, supplies its extraction serial to the
dedup gate through the registry, and names itself in reports and in the
language a chunk's identity is keyed by.
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration

PLAINTEXT = "plaintext"
"""Id of the fallback language."""

plaintext = LanguageRegistration(id=PLAINTEXT)

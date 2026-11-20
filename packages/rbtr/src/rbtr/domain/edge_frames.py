"""Resolution policy and the frames edge inference exchanges.

Plugins declare how each language's imports name files and sections;
`languages.resolution` compiles those declarations into one
`LanguageResolutionRow` per language. The index reads located chunks into
`EdgeChunkRow` and stores `InferredEdgeRow`. Between
them, `domain.edges` and `domain.resolution` exchange `ImportRow`,
`FileMatchRow`, and the module lookups `ModulePathRow` and
`ModuleFileRow`.
"""

from __future__ import annotations

from enum import StrEnum

import dataframely as dy

from rbtr.domain.models import ChunkKind, EdgeKind


class ModuleStyle(StrEnum):
    """How a language writes a module: as a path (`./b`, `b.css`), dotted (`pkg.b`) or scoped (`pkg::b`)."""

    PATH = "path"
    DOTTED = "dotted"
    SCOPED = "scoped"


class SectionNames(StrEnum):
    """How an import compares with a section's name.

    `EXACT` compares the name as written (an HTML `id`). `FOLDED` ignores
    case, accents, punctuation and leading digits, as renderers do when
    they make a heading's anchor (`Getting Started` → `#getting-started`).
    """

    EXACT = "exact"
    FOLDED = "folded"


class ImportWay(StrEnum):
    """How an import reaches what it names.

    `MODULE`: it names a module or a relative depth, resolved to files.
    `NAME`: prose names something and no file (an rST role). `TEXT`: its
    extractor recorded neither, so its words are matched against file stems.
    """

    MODULE = "module"
    NAME = "name"
    TEXT = "text"


class LanguageResolutionRow(dy.Schema):
    """How imports written in one language name files and sections.

    The list columns are in lookup order: the first matching source root,
    substitution, extension or index file wins. `extensions` and
    `index_files` are those of the languages the imports may reach;
    `own_extensions` are the writing language's own, preferred when two
    files differ only by extension. `parent_module` and `current_module`
    are the words a module starts with to climb one level or stay in the
    importing module, as Rust's `super` and `self`, null for none;
    `directory_module_files` the file names that are their directory's
    module.
    """

    language = dy.String(primary_key=True)
    extensions = dy.List(dy.String(nullable=False), nullable=False)
    index_files = dy.List(dy.String(nullable=False), nullable=False)
    source_roots = dy.List(dy.String(nullable=False), nullable=False)
    substitutions = dy.List(
        dy.Struct({"prefix": dy.String(nullable=False), "replacement": dy.String(nullable=False)}),
        nullable=False,
    )
    module_style = dy.Enum(s.value for s in ModuleStyle)
    own_extensions = dy.List(dy.String(nullable=False), nullable=False)
    package_directory = dy.Bool(nullable=False)
    prose = dy.Bool(nullable=False)
    section_names = dy.Enum(s.value for s in SectionNames)
    parent_module = dy.String(nullable=True)
    current_module = dy.String(nullable=True)
    directory_module_files = dy.List(dy.String(nullable=False), nullable=False)


class EdgeChunkRow(dy.Schema):
    """A located chunk as edge inference reads it.

    Module, names, hint and bindings are null on all but an import;
    `refs` is null on a role that records none.
    """

    id = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    language = dy.String(nullable=False)
    module = dy.String(nullable=True)
    names = dy.String(nullable=True)
    language_hint = dy.String(nullable=True)


class ImportRow(dy.Schema):
    """An import at its source location, normalised, before any file is resolved.

    `text` is the import chunk's own text, which `TEXT` imports match
    against file stems. Its module is read once, by its writing language's
    style: its relative prefix (`./`, `..`, `super::`) is `depth`, the
    levels it climbs from the importing file, 0 when it climbs none, and
    `module` is the rest.
    """

    source_id = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    source_language = dy.String(nullable=False)
    text = dy.String(nullable=False)
    way = dy.Enum(w.value for w in ImportWay)
    kind = dy.Enum(k.value for k in EdgeKind)
    depth = dy.Int64(nullable=False, min=0)
    module = dy.String(nullable=False)
    names = dy.List(dy.String(nullable=False), nullable=False)
    language_hint = dy.String(nullable=False)


class ModulePathRow(dy.Schema):
    """One attempt of a module lookup: a path, looked up by one language's resolution.

    A lookup's attempts are tried in order and the first that finds a file
    wins. `importer_extension` is the referring file's extension, empty
    when nothing refers.
    """

    lookup = dy.UInt32(nullable=False)
    attempt = dy.UInt32(nullable=False)
    language = dy.String(nullable=False)
    path = dy.String(nullable=False)
    importer_extension = dy.String(nullable=False)


class ModuleFileRow(dy.Schema):
    """A file one lookup attempt names; a package directory's attempt names several."""

    lookup = dy.UInt32(nullable=False)
    attempt = dy.UInt32(nullable=False)
    target_path = dy.String(nullable=False)


class FileMatchRow(dy.Schema):
    """One file an import at a source location reaches."""

    source_id = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    target_path = dy.String(nullable=False)


class InferredEdgeRow(dy.Schema):
    """An import, call or base at one location reaching a chunk at another; both paths are part of it."""

    source_id = dy.String(nullable=False)
    target_id = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)
    source_path = dy.String(nullable=False)
    target_path = dy.String(nullable=False)

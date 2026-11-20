"""Resolution policy and the frames edge inference exchanges.

Plugins declare how each language's imports name files and sections;
`languages.resolution` compiles those declarations into one
`LanguageResolutionRow` per language. The index reads located chunks into
`EdgeChunkRow` and stores `InferredEdgeRow`. Between
them, `domain.edges` and `domain.resolution` exchange `ImportRow`,
`FileMatchRow`, and the module lookups `ModulePathRow` and
`ModuleFileRow`, which qualified-name lookup also uses. `domain.calls`
resolves calls and bases from `ExportRow` (what each file makes available)
and `PathRow` (what each chunk calls or extends).
"""

from __future__ import annotations

from enum import StrEnum

import dataframely as dy
import polars as pl

from rbtr.domain.models import ChunkKind, EdgeKind, Lookup
from rbtr.domain.rules import ChunkRow


class ModuleStyle(StrEnum):
    """How a language writes a module: as a path (`./b`, `b.css`), dotted (`pkg.b`) or scoped (`pkg::b`)."""

    PATH = "path"
    DOTTED = "dotted"
    SCOPED = "scoped"


class MemberOrder(StrEnum):
    """The order a language searches a class's bases for a member it does not define.

    `LISTED` and `REVERSED` walk the bases depth first, keeping each
    ancestor's last visit, so a base shared by two comes after both:
    `LISTED` takes them as written (Python, Java, C++, JavaScript, Rust),
    `REVERSED` last written first (Ruby, whose latest `include` precedes
    earlier ones and its superclass). `NEAREST` takes the shallowest, ties
    alike (Go's embedding).
    """

    LISTED = "listed"
    REVERSED = "reversed"
    NEAREST = "nearest"


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
    members_in_scope = dy.Bool(nullable=False)
    directory_scope = dy.Bool(nullable=False)
    linked_functions = dy.Bool(nullable=False)
    member_order = dy.Enum(o.value for o in MemberOrder)
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
    scope = dy.String(nullable=False)
    language = dy.String(nullable=False)
    module = dy.String(nullable=True)
    names = dy.String(nullable=True)
    language_hint = dy.String(nullable=True)
    bindings = ChunkRow.metadata.inner["bindings"]
    refs = ChunkRow.metadata.inner["refs"]


class ImportRow(dy.Schema):
    """An import at its source location, normalised, before any file is resolved.

    `text` is the import chunk's own text, which `TEXT` imports match
    against file stems. `written` is its module as written, read once by
    its writing language's style: its relative prefix (`./`, `..`,
    `super::`) is `depth`, the levels it climbs from the importing file, 0
    when it climbs none, and `module` is the rest. `names` lists what it
    names, `*` for every name; `bindings` are the names it binds, a
    module binding named as `written`.
    """

    source_id = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    source_language = dy.String(nullable=False)
    text = dy.String(nullable=False)
    way = dy.Enum(w.value for w in ImportWay)
    kind = dy.Enum(k.value for k in EdgeKind)
    line_start = dy.Int32(nullable=False)
    written = dy.String(nullable=False)
    depth = dy.Int64(nullable=False, min=0)
    module = dy.String(nullable=False)
    names = dy.List(dy.String(nullable=False), nullable=False)
    bindings = dy.List(
        dy.Struct({"name": dy.String(nullable=False), "local": dy.String(nullable=False)}),
        nullable=False,
    )
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


class ExportRow(dy.Schema):
    """A name a file makes available at a `::` scope, and what it reaches: a definition at a location, or a module's file.

    One of `def_id` (with `def_path`) and `module_path` is set.
    """

    file_path = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    name = dy.String(nullable=False)
    def_id = dy.String(nullable=True)
    def_path = dy.String(nullable=True)
    module_path = dy.String(nullable=True)

    @dy.rule()
    def reaches_one_definition_or_module(cls) -> pl.Expr:
        return (cls.def_id.col.is_null() == cls.def_path.col.is_null()) & (
            cls.def_id.col.is_null() != cls.module_path.col.is_null()
        )


class PathRow(dy.Schema):
    """A call or base to resolve, at its source location: the edge it makes, its lookup, path and chain.

    `caller_chain` is the `::` scope its head is looked up from;
    `position` its place among its chunk's calls or bases, from 0.
    """

    source_id = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    language = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)
    lookup = dy.Enum(r.value for r in Lookup)
    path = dy.List(dy.String(nullable=False), nullable=False)
    caller_chain = dy.String(nullable=False)
    position = dy.Int32(nullable=False)

    @dy.rule()
    def bases_are_unqualified_paths(cls) -> pl.Expr:
        return (cls.kind.col != EdgeKind.INHERITS.value) | (
            cls.lookup.col == Lookup.UNQUALIFIED.value
        )


class ResolvedPathRow(PathRow):
    """A path and a definition it reaches, at that definition's location.

    Its edge is its source, target and kind; a base's `position` orders
    the class's bases for the member order.
    """

    target_id = dy.String(nullable=False)
    target_path = dy.String(nullable=False)

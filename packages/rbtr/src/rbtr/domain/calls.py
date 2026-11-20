"""Call and inherits edges: the definitions each chunk's calls and bases name.

`called_targets` is edge inference's fourth step. Its runbook:

1. `exports`: the names every file makes available, each reaching a
   definition or a module (`ExportRow`).
2. `paths`: every base and every call extraction recorded, at each
   location its chunk sits (`PathRow`).
3. `path_targets` on the bases: each base with the classes it reaches
   (`ResolvedPathRow`).
4. `path_targets` on the calls, through those bases in member order;
   both are projected to `inherits` and `calls` edges.

Re-exports, star imports and bases are followed at most `steps` times.
The functions returning `pl.Expr` are the column calculations the steps
use.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.domain.edge_frames import (
    EdgeChunkRow,
    ExportRow,
    FileMatchRow,
    ImportRow,
    ImportWay,
    InferredEdgeRow,
    LanguageResolutionRow,
    MemberOrder,
    ModuleStyle,
    PathRow,
    ResolvedPathRow,
)
from rbtr.domain.models import ChunkKind, EdgeKind, Lookup
from rbtr.domain.resolution import (
    parent_directory,
    resolved_files,
)
from rbtr.domain.rules import DEFINITION_KINDS

TARGET_KINDS = [ChunkKind.FUNCTION, ChunkKind.METHOD, ChunkKind.CLASS, ChunkKind.TEST_FUNCTION]
"""What a call or base can reach."""


def chain(scope: pl.Expr, name: pl.Expr) -> pl.Expr:
    """The `::` address of a definition named *name* in *scope*: `Order::save`."""
    return pl.when(scope == "").then(name).otherwise(scope + "::" + name)


def chain_parts(column: str) -> pl.Expr:
    """The names of the `::` chain in *column*: `A::b` gives `["A", "b"]`, an empty chain none."""
    return pl.col(column).str.split("::").list.eval(pl.element().filter(pl.element() != ""))


def chain_depths() -> pl.Expr:
    """Each depth from 0 to the number of `parts`, one per prefix of the chain."""
    return pl.int_ranges(0, pl.col("parts").list.len() + 1)


def chain_prefix() -> pl.Expr:
    """The chain's prefix of `depth` names: at depth 1 of `A::b`, `A`."""
    return pl.col("parts").list.head(pl.col("depth")).list.join("::")


def listed_binding() -> pl.Expr:
    """Whether a binding's `name` is one its import lists, rather than a module."""
    return pl.col("names").list.contains(pl.col("name"))


def bound_module() -> pl.Expr:
    """The module a binding looks up: for a listed name, the submodule of that name; else its `name`.

    `from pkg import mod` looks up `pkg.mod` (`pkg/mod` for a path
    language), and `from . import mod` looks up `mod` at the import's
    depth (`bound_depth`); `import a.b` binding `a` looks up `a`.
    """
    return (
        pl.when(~listed_binding())
        .then("name")
        .when(pl.col("module") == "")
        .then("name")
        .when(pl.col("module_style") == ModuleStyle.DOTTED.value)
        .then(pl.col("module") + "." + pl.col("name"))
        .when(pl.col("module_style") == ModuleStyle.SCOPED.value)
        .then(pl.col("module") + "::" + pl.col("name"))
        .otherwise(pl.col("module") + "/" + pl.col("name"))
    )


def bound_depth() -> pl.Expr:
    """The relative depth of `bound_module`: its import's for a listed name, else 0, a name from the root."""
    return pl.when(listed_binding()).then("depth").otherwise(0)


def module_lookup(module: str, depth: str) -> dict[str, pl.Expr | str]:
    """`ImportRow`'s columns for looking up *module* at *depth* from `source_path`, keyed by `key`."""
    return {
        "source_id": pl.col("key").cast(pl.String),
        "source_path": "source_path",
        "source_language": "language",
        "text": pl.lit(""),
        "way": pl.lit(ImportWay.MODULE.value),
        "kind": pl.lit(EdgeKind.IMPORTS.value),
        "line_start": "line_start",
        "written": module,
        "depth": depth,
        "module": module,
        "names": pl.lit([], dtype=pl.List(pl.String)),
        "bindings": pl.lit([], dtype=ImportRow.bindings.dtype),
        "language_hint": pl.lit(""),
    }


def exports(
    imports: dy.DataFrame[ImportRow],
    chunks: dy.DataFrame[EdgeChunkRow],
    files: dy.DataFrame[FileMatchRow],
    repo_files: set[str],
    resolutions: dy.DataFrame[LanguageResolutionRow],
    steps: int,
) -> dy.DataFrame[ExportRow]:
    """The names each file makes available, each at a scope, and the definition or module it reaches.

    A file makes available its own functions and classes, each at its
    scope (a C++ class in `namespace shop` at `shop`), and, at the top
    level, every name its *imports* bind (`requested_imports`; *files* are
    the files each reaches):

    - a module (`import os`, `import * as utils`), bound by a name the
      import does not list, reaches that module's files;
    - a listed name (`from m import f`, `f as g`) reaches what the imported
      files make available under it at their top level, else the submodule
      of that name (`from pkg import mod`);
    - a listed name written as a path (`Strings.shout`) reaches the member
      at that path in the imported files;
    - a star import (`from m import *`, `#include`) makes available every
      name its files do, at its scope, the first such import winning.

    A definition wins over
    an import of the same name, and a listed name over a star import. Names made available through other files'
    imports (re-exports) are followed at most *steps* times. A file of a
    language with `directory_scope` also makes available, last, the
    top-level functions and classes of its directory's other files in
    that language.
    """
    location = ["source_id", "source_path"]
    module_imports = (
        imports.filter(pl.col("way") == ImportWay.MODULE)
        .select(
            "source_id",
            "source_path",
            "line_start",
            "written",
            "depth",
            "module",
            "names",
            "bindings",
            language="source_language",
        )
        .join(resolutions.select("language", "module_style"), on="language", how="left")
        .join(
            files.group_by(location).agg(files=pl.col("target_path").unique().sort()),
            on=location,
            how="left",
        )
        .with_columns(pl.col("files").fill_null([]))
    )
    bound = (
        module_imports.explode("bindings")
        .drop_nulls("bindings")
        .with_columns(
            name=pl.col("bindings").struct.field("name"),
            local=pl.col("bindings").struct.field("local"),
        )
        .with_row_index("key")
        .with_columns(lookup=bound_module(), lookup_depth=bound_depth())
    )
    looked_up = (
        resolved_files(
            bound.select(**module_lookup("lookup", "lookup_depth")).pipe(
                ImportRow.validate, cast=True
            ),
            repo_files,
            resolutions,
        )
        .group_by("source_id")
        .agg(found=pl.col("target_path").unique().sort())
        .select(key=pl.col("source_id").cast(pl.UInt32), found="found")
    )
    bound = bound.join(looked_up, on="key", how="left").with_columns(pl.col("found").fill_null([]))
    listed = listed_binding()
    no_scope = pl.lit("")
    definitions = chunks.filter(
        pl.col("kind").is_in(TARGET_KINDS)
        & (pl.col("kind") != ChunkKind.METHOD)
        & (pl.col("name") != "")
    ).select(
        "file_path",
        "scope",
        "name",
        def_id="id",
        def_path="file_path",
        module_path=pl.lit(None, pl.String),
    )
    modules = (
        bound.filter(~listed)
        .select(
            file_path="source_path",
            scope=no_scope,
            name="local",
            def_id=pl.lit(None, pl.String),
            def_path=pl.lit(None, pl.String),
            module_path=pl.when(pl.col("name") == pl.col("written"))
            .then("files")
            .otherwise("found"),
        )
        .explode("module_path")
        .drop_nulls("module_path")
    )
    names = bound.filter(listed).select(
        file_path="source_path", local="local", name="name", files="files", submodule="found"
    )
    # A listed name written as a path (`Strings.shout`, Java's static import)
    # is a member: its last name inside the scope its others make.
    parts = (
        pl.when(pl.col("module_style") == ModuleStyle.DOTTED)
        .then(pl.col("name").str.split("."))
        .when(pl.col("module_style") == ModuleStyle.SCOPED)
        .then(pl.col("name").str.split("::"))
    )
    members = (
        bound.filter(listed)
        .with_columns(parts=parts)
        .filter(pl.col("parts").list.len() > 1)
        .with_columns(
            member_scope=pl.col("parts").list.head(pl.col("parts").list.len() - 1).list.join("::"),
            member=pl.col("parts").list.last(),
        )
        .explode("files")
        .join(
            chunks.filter(pl.col("kind").is_in(TARGET_KINDS)).select(
                "file_path", "scope", "name", def_id="id"
            ),
            left_on=["files", "member_scope", "member"],
            right_on=["file_path", "scope", "name"],
        )
        .select(
            file_path="source_path",
            scope=no_scope,
            name="local",
            def_id="def_id",
            def_path="files",
            module_path=pl.lit(None, pl.String),
        )
    )
    stars = module_imports.filter(pl.col("names").list.contains("*")).select(
        file_path="source_path", files="files", rank="line_start"
    )
    key = ["file_path", "scope", "name"]
    known = pl.concat([definitions, modules, members]).unique()
    for _ in range(steps):
        through_names = (
            names.explode("files")
            .join(
                known.filter(pl.col("scope") == ""),
                left_on=["files", "name"],
                right_on=["file_path", "name"],
            )
            .select("file_path", "scope", "def_id", "def_path", "module_path", name="local")
            .select(known.columns)
        )
        submodules = (
            names.join(
                through_names.select("file_path", local="name"),
                on=["file_path", "local"],
                how="anti",
            )
            .explode("submodule")
            .drop_nulls("submodule")
            .select(
                "file_path",
                scope=no_scope,
                name="local",
                def_id=pl.lit(None, pl.String),
                def_path=pl.lit(None, pl.String),
                module_path="submodule",
            )
            .select(known.columns)
        )
        through_stars = (
            stars.explode("files")
            .join(known, left_on="files", right_on="file_path")
            .filter(pl.col("rank") == pl.col("rank").min().over("file_path", "scope", "name"))
            .select(known.columns)
            .join(through_names, on=key, how="anti")
        )
        new = (
            pl.concat([through_names, submodules, through_stars])
            .join(known, on=key, how="anti")
            .unique()
        )
        if new.is_empty():
            break
        known = pl.concat([known, new])
    scoped = resolutions.filter(pl.col("directory_scope")).select("language")
    siblings = (
        chunks.join(scoped, on="language")
        .select("file_path", "language", directory=parent_directory(pl.col("file_path")))
        .unique()
        .join(
            definitions.join(
                chunks.select(def_id="id", def_path="file_path", language="language"),
                on=["def_id", "def_path"],
            )
            .join(scoped, on="language")
            .drop("file_path")
            .with_columns(directory=parent_directory(pl.col("def_path"))),
            on=["directory", "language"],
        )
        .filter(pl.col("def_path") != pl.col("file_path"))
        .select(known.columns)
        .join(known, on=key, how="anti")
    )
    return pl.concat([known, siblings]).unique().pipe(ExportRow.validate, cast=True)


def paths(chunks: dy.DataFrame[EdgeChunkRow], kind: EdgeKind) -> dy.DataFrame[PathRow]:
    """Every base (`INHERITS`) or call (`CALLS`) of *chunks*, at each location its chunk sits.

    A call is looked up from its chunk's own chain when the chunk is a
    definition, else from its scope; a base from its class's scope. A call
    on a local (`TYPE` lookup) is on an instance of the local's written
    type, so it is looked up as that type's path followed by the rest:
    `order.save` with `order: Order` is `Order` → `save`. One with no
    written type, or calling the value itself (`order()`), is left out.
    """
    own_chain = pl.col("kind").is_in(DEFINITION_KINDS) & pl.lit(kind == EdgeKind.CALLS)
    ref = pl.element().struct
    of_kind = pl.col("refs").list.filter(
        (ref.field("kind") == kind.value)
        & (
            (ref.field("lookup") != Lookup.TYPE.value)
            | (ref.field("type").is_not_null() & (ref.field("path").list.len() > 1))
        )
    )
    on_value = pl.col("refs").struct.field("lookup") == Lookup.TYPE.value
    return (
        chunks.with_columns(refs=of_kind)
        .filter(pl.col("refs").list.len() > 0)
        .with_columns(
            caller_chain=pl.when(own_chain)
            .then(chain(pl.col("scope"), pl.col("name")))
            .otherwise(pl.col("scope")),
            position=pl.int_ranges(pl.col("refs").list.len()),
        )
        .explode("refs", "position")
        .select(
            "language",
            source_id="id",
            source_path="file_path",
            kind=pl.lit(kind.value),
            lookup=pl.when(on_value)
            .then(pl.lit(Lookup.UNQUALIFIED.value))
            .otherwise(pl.col("refs").struct.field("lookup")),
            path=pl.when(on_value)
            .then(
                pl.col("refs")
                .struct.field("type")
                .list.concat(pl.col("refs").struct.field("path").list.slice(1))
            )
            .otherwise(pl.col("refs").struct.field("path")),
            caller_chain="caller_chain",
            position="position",
        )
        .pipe(PathRow.validate, cast=True)
    )


def path_targets(
    paths_to_resolve: dy.DataFrame[PathRow],
    chunks: dy.DataFrame[EdgeChunkRow],
    names: dy.DataFrame[ExportRow],
    bases: dy.DataFrame[ResolvedPathRow],
    resolutions: dy.DataFrame[LanguageResolutionRow],
    steps: int,
) -> dy.DataFrame[ResolvedPathRow]:
    """Each path with each definition it reaches; an `INHERITS` path reaches classes only.

    The head is resolved first:

    - `UNQUALIFIED`: what the source's file makes available under that
      name (*names*: its own functions and classes and what it imports) at
      the longest prefix of the path's chain, so an inner definition wins
      and a method is never in scope; else, for a language with
      `members_in_scope`, the member of that name of the nearest
      enclosing class or its bases;
    - `MEMBER`: the source's nearest enclosing class, among those its file
      sees (an out-of-class C++ member's class is in a header); `BASE`:
      the same, looked up from its bases.

    Each further name is looked up in what the last reached: a module's
    name (*names*), or a class's member, else the member of its first
    ancestor holding one, in the language's `MemberOrder` (the resolved
    *bases*, followed at most *steps* times): `LISTED` and `REVERSED` order them by
    each one's last walk through the bases' positions, `NEAREST` by depth,
    ties alike. A function ends the path. A member reached also reaches
    each definition of it outside its class (C++'s `bool Order::check()
    {…}`): a function scoped by the class's chain, named like the member.
    A function reached in a language with `linked_functions` also reaches
    each function of its scope and name in those languages (C's prototype
    and definition). No match makes no edge; several make an edge to each.
    """
    asks = paths_to_resolve.with_row_index("key")
    defs = chunks.filter(pl.col("kind").is_in(TARGET_KINDS) & (pl.col("name") != "")).select(
        def_id="id",
        def_path="file_path",
        def_kind=pl.col("kind").cast(pl.String),
        def_name="name",
        def_scope="scope",
        def_chain=chain(pl.col("scope"), pl.col("name")),
        def_language="language",
    )
    classes = defs.filter(pl.col("def_kind") == ChunkKind.CLASS)
    # Each class with itself at depth 0, then every walk through its bases by
    # their positions, negated where the language searches them last first.
    level = classes.join(
        resolutions.select(def_language="language", member_order="member_order"), on="def_language"
    ).select(
        class_id="def_id",
        class_path="def_path",
        ancestor_id="def_id",
        ancestor_path="def_path",
        depth=pl.lit(0),
        walk=pl.lit([], dtype=pl.List(pl.Int32)),
        member_order="member_order",
    )
    lineage = [level]
    for depth in range(1, steps + 1):
        level = (
            level.join(
                bases.select(
                    ancestor_id="source_id",
                    ancestor_path="source_path",
                    base_id="target_id",
                    base_path="target_path",
                    position="position",
                ),
                on=["ancestor_id", "ancestor_path"],
            )
            .select(
                "class_id",
                "class_path",
                ancestor_id="base_id",
                ancestor_path="base_path",
                depth=pl.lit(depth),
                walk=pl.col("walk").list.concat(
                    pl.when(pl.col("member_order") == MemberOrder.REVERSED)
                    .then(-pl.col("position"))
                    .otherwise(pl.col("position"))
                    .cast(pl.Int32)
                ),
                member_order="member_order",
            )
            .unique()
        )
        if level.is_empty():
            break
        lineage.append(level)
    # An ancestor's rank in its class's search: its last walk in order, or its
    # depth where the nearest wins, ties alike.
    by_class = ["class_id", "class_path"]
    ancestors = (
        pl.concat(lineage)
        .sort("walk")
        .group_by(*by_class, "ancestor_id", "ancestor_path")
        .agg(pl.col("walk").last(), pl.col("depth").min(), pl.col("member_order").first())
        .sort(*by_class, "walk")
        .with_columns(
            rank=pl.when(pl.col("member_order") == MemberOrder.NEAREST)
            .then(pl.col("depth"))
            .otherwise(pl.int_range(pl.len()).over(by_class))
        )
        .join(
            classes.select("def_id", "def_path", ancestor_chain="def_chain"),
            left_on=["ancestor_id", "ancestor_path"],
            right_on=["def_id", "def_path"],
        )
    )
    # The head.
    named = asks.filter(pl.col("lookup") == Lookup.UNQUALIFIED).with_columns(
        head=pl.col("path").list.first(), parts=chain_parts("caller_chain")
    )
    # The innermost scope holding the head's name decides it.
    available = (
        named.with_columns(depth=chain_depths())
        .explode("depth")
        .with_columns(prefix=chain_prefix())
        .join(
            names,
            left_on=["source_path", "prefix", "head"],
            right_on=["file_path", "scope", "name"],
        )
        .filter(pl.col("depth") == pl.col("depth").max().over("key"))
        .select("key", "def_id", "def_path", "module_path", min_depth=pl.lit(0))
    )
    in_class = (
        named.join(available.select("key").unique(), on="key", how="anti")
        .join(resolutions.filter(pl.col("members_in_scope")).select("language"), on="language")
        .select("key", "source_path", "caller_chain", lookup=pl.lit(Lookup.MEMBER.value))
    )
    # The classes a file sees: its own, and those its names reach.
    visible_classes = names.join(
        classes.select("def_id", "def_path", "def_chain"), on=["def_id", "def_path"]
    ).select("file_path", "def_id", "def_path", "def_chain")
    enclosing = (
        pl.concat(
            [
                asks.filter(pl.col("lookup") != Lookup.UNQUALIFIED).select(
                    "key", "source_path", "caller_chain", pl.col("lookup").cast(pl.String)
                ),
                in_class,
            ]
        )
        .with_columns(parts=chain_parts("caller_chain"))
        .with_columns(depth=chain_depths())
        .explode("depth")
        .filter(pl.col("depth") > 0)
        .with_columns(prefix=chain_prefix())
        .join(
            visible_classes,
            left_on=["source_path", "prefix"],
            right_on=["file_path", "def_chain"],
        )
        .filter(pl.col("depth") == pl.col("depth").max().over("key"))
        .select(
            "key",
            "def_id",
            "def_path",
            module_path=pl.lit(None, pl.String),
            min_depth=pl.when(pl.col("lookup") == Lookup.BASE).then(1).otherwise(0),
        )
    )
    current = pl.concat([available, enclosing], how="vertical_relaxed")

    # Each further name; a path whose head was looked up in its class starts at the head.
    start = (
        pl.when(
            (pl.col("lookup") == Lookup.UNQUALIFIED)
            & ~pl.col("key").is_in(in_class["key"].implode())
        )
        .then(1)
        .otherwise(0)
    )
    progress = asks.select("key", "path", start=start)
    found = [pl.DataFrame(schema={"key": pl.UInt32, "def_id": pl.String, "def_path": pl.String})]
    for step in range(steps):
        if current.is_empty():
            break
        at = current.join(progress, on="key").with_columns(position=pl.col("start") + step)
        found.append(
            at.filter(
                (pl.col("position") >= pl.col("path").list.len()) & pl.col("def_id").is_not_null()
            ).select("key", "def_id", "def_path")
        )
        moving = at.filter(pl.col("position") < pl.col("path").list.len()).with_columns(
            part=pl.col("path").list.get(pl.col("position"))
        )
        in_modules = moving.join(
            names.filter(pl.col("scope") == ""),
            left_on=["module_path", "part"],
            right_on=["file_path", "name"],
            suffix="_next",
        ).select(
            "key",
            def_id="def_id_next",
            def_path="def_path_next",
            module_path="module_path_next",
            min_depth=pl.lit(0),
        )
        in_classes = (
            moving.join(
                ancestors, left_on=["def_id", "def_path"], right_on=["class_id", "class_path"]
            )
            .filter(pl.col("depth") >= pl.col("min_depth"))
            .join(
                defs.select("def_path", "def_scope", "def_name", member_id="def_id"),
                left_on=["ancestor_path", "ancestor_chain", "part"],
                right_on=["def_path", "def_scope", "def_name"],
            )
            .filter(pl.col("rank") == pl.col("rank").min().over("key"))
            .select(
                "key",
                def_id="member_id",
                def_path="ancestor_path",
                module_path=pl.lit(None, pl.String),
                min_depth=pl.lit(0),
            )
        )
        current = pl.concat([in_modules, in_classes], how="vertical_relaxed")
    # A member reached in its class also reaches each definition of it outside
    # the class: a function scoped by the class's chain, named like the member.
    reached = pl.concat(found, how="vertical_relaxed")
    outside = (
        reached.join(
            defs.filter(pl.col("def_kind") == ChunkKind.METHOD).select(
                "def_id", "def_path", "def_scope", "def_name"
            ),
            on=["def_id", "def_path"],
        )
        .join(
            defs.filter(pl.col("def_kind") == ChunkKind.FUNCTION).select(
                "def_scope", "def_name", outside_id="def_id", outside_path="def_path"
            ),
            on=["def_scope", "def_name"],
        )
        .select("key", def_id="outside_id", def_path="outside_path")
    )
    # Where functions are linked, a function reached reaches each of its
    # scope and name: a header's prototype and every definition.
    linked_functions = (
        defs.filter(pl.col("def_kind") == ChunkKind.FUNCTION)
        .join(
            resolutions.filter(pl.col("linked_functions")).select(def_language="language"),
            on="def_language",
        )
        .select("def_id", "def_path", "def_scope", "def_name")
    )
    linked = (
        pl.concat([reached, outside], how="vertical_relaxed")
        .join(linked_functions, on=["def_id", "def_path"])
        .join(
            linked_functions.select(
                "def_scope", "def_name", linked_id="def_id", linked_path="def_path"
            ),
            on=["def_scope", "def_name"],
        )
        .select("key", def_id="linked_id", def_path="linked_path")
    )
    return (
        pl.concat([reached, outside, linked], how="vertical_relaxed")
        .join(asks, on="key")
        .join(defs.select("def_id", "def_path", "def_kind"), on=["def_id", "def_path"])
        .filter((pl.col("kind") != EdgeKind.INHERITS) | (pl.col("def_kind") == ChunkKind.CLASS))
        .select(*PathRow.column_names(), target_id="def_id", target_path="def_path")
        .unique()
        .pipe(ResolvedPathRow.validate, cast=True)
    )


def called_targets(
    imports: dy.DataFrame[ImportRow],
    chunks: dy.DataFrame[EdgeChunkRow],
    files: dy.DataFrame[FileMatchRow],
    repo_files: set[str],
    resolutions: dy.DataFrame[LanguageResolutionRow],
    steps: int,
) -> dy.DataFrame[InferredEdgeRow]:
    """The `inherits` edges every base makes and the `calls` edges every call makes.

    *imports* are the snapshot's imports (`edges.requested_imports`) and
    *files* the files each reaches (`resolution.resolved_files`); chains
    are followed at most *steps* times. The module docstring lists
    the steps.
    """
    names = exports(imports, chunks, files, repo_files, resolutions, steps)
    bases = path_targets(
        paths(chunks, EdgeKind.INHERITS),
        chunks,
        names,
        ResolvedPathRow.create_empty(),
        resolutions,
        steps,
    )
    calls = path_targets(paths(chunks, EdgeKind.CALLS), chunks, names, bases, resolutions, steps)
    return (
        pl.concat([bases, calls])
        .select(InferredEdgeRow.column_names())
        .unique()
        .pipe(InferredEdgeRow.validate, cast=True)
    )

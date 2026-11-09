"""Module resolution: the repository files an import's module names.

Each language plugin declares on its `LanguageRegistration` how its
references name files: the extensions and index files of the languages
it imports, its source roots, its path substitutions (and a manifest
that adds more), its module style, and whether a reference may name a
directory.  `build_resolution_map` gathers those declarations per
language into `ImportResolution`s; `import_files` and `module_files`
apply them.  Nothing here knows a language by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from rbtr.domain.models import ImportMeta
from rbtr.languages.manager import LanguageManager
from rbtr.languages.registration import ModuleStyle, parse_path_relative


@dataclass(frozen=True)
class ImportResolution:
    """How imports written in one language resolve, as its plugin declares.

    Built by `build_resolution_map` from the registrations and passed
    to resolution and edge inference as plain data.
    """

    extensions: tuple[str, ...]
    """File extensions to try (collected from target languages)."""
    index_files: tuple[str, ...]
    """Directory entry points to try (collected from target languages)."""
    source_roots: tuple[str, ...]
    """Directories to prepend when resolving absolute imports."""
    path_substitutions: tuple[tuple[str, str], ...]
    """Module path prefix replacements (e.g. `("crate/", "src/")`)."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How module strings map to file paths. See `ModuleStyle`."""
    own_extensions: frozenset[str] = frozenset()
    """The importing language's *own* extensions (distinct from the flattened
    target `extensions`). Used to break same-path ties in the suffix tier."""
    package_directory: bool = False
    """Whether a reference may name a directory whose every file it reaches."""
    prose: bool = False
    """Whether the language's references document what they name (see
    `LanguageRegistration.prose`)."""


def build_resolution_map(
    mgr: LanguageManager,
    manifests: dict[str, str] | None = None,
) -> dict[str, ImportResolution]:
    """Build language → resolution hints from plugin registrations.

    *manifests* maps a repository file's path to its text, for the files
    languages declare as their `manifest`. A language whose manifest is
    present gains the substitutions it reads from it, ahead of the ones
    its registration declares: Go's `go.mod` names the module prefix its
    imports carry, which is a fact about the repository rather than about
    the language.
    """
    manifests = manifests or {}
    result: dict[str, ImportResolution] = {}
    for lang_id in mgr.all_language_ids():
        reg = mgr.get_registration(lang_id)
        if reg is None:
            continue
        from_manifest: tuple[tuple[str, str], ...] = ()
        if reg.manifest and (text := manifests.get(reg.manifest)) is not None:
            from_manifest = reg.substitutions_from_manifest(text)
        target_langs = reg.import_targets or frozenset({lang_id})
        exts: list[str] = []
        idxs: list[str] = []
        for target in target_langs:
            target_reg = mgr.get_registration(target)
            if target_reg is not None:
                exts.extend(target_reg.extensions)
                idxs.extend(target_reg.index_files)
        result[lang_id] = ImportResolution(
            extensions=tuple(exts),
            index_files=tuple(idxs),
            source_roots=reg.source_roots,
            path_substitutions=from_manifest + reg.path_substitutions,
            module_style=reg.module_style,
            own_extensions=reg.extensions,
            package_directory=reg.package_directory,
            prose=reg.prose,
        )
    return result


def import_files(
    meta: ImportMeta,
    file_path: str,
    language: str,
    repo_files: set[str],
    resolution_map: Mapping[str, ImportResolution],
) -> list[str]:
    """The files the import *meta*, written in *language* at *file_path*, names.

    The rules are those of *meta*'s `language_hint` when it has one (an
    HTML `<script src>` names JavaScript), else those of *language*; a
    language with none resolves nothing.  The module is looked for:

    1. Anchored to the importer, when the reference carries its own
       depth: the extractor's `dots`, or a path-style `./` or `../`.
    2. Otherwise from the repository root (`module_files`).
    3. For a path-style reference found nowhere from the root, beside
       the importer: `@import "b.css"` in `css/a.css` means `css/b.css`.

    A file never resolves to itself, however its neighbours are named.
    """
    resolution = resolution_map.get(meta.language_hint or language)
    if resolution is None:
        return []
    importer_ext = PurePosixPath(file_path).suffix
    module, dots = meta.module, int(meta.dots) if meta.dots else 0
    if not dots and resolution.module_style is ModuleStyle.PATH:
        dots, module = parse_path_relative(module)
    if dots:
        anchored = _climb(file_path, dots, module)
        found = (
            [] if anchored is None else module_files(anchored, repo_files, resolution, importer_ext)
        )
    elif not module:
        found = []
    else:
        found = module_files(module, repo_files, resolution, importer_ext)
        if not found and resolution.module_style is ModuleStyle.PATH:
            beside = _climb(file_path, 1, module)
            if beside is not None and beside != module:
                found = module_files(beside, repo_files, resolution, importer_ext)
    # A file does not import itself whole; it may name something in itself.
    return [f for f in found if meta.names or f != file_path]


def module_files(
    path: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str = "",
) -> list[str]:
    """The files the module *path* names: one file, or a directory's.

    *path* is `/`-separated, or dotted for a dotted module style.  After
    the language's path substitutions (`_substituted`), it is looked for:

    1. Under each source root: as is, with each extension, then as a
       directory's index file (`_under_source_roots`).
    2. As a full-path suffix anywhere, ambiguity resolving nothing
       (`_by_path_suffix`); *importer_ext* breaks ties between files
       that differ only by extension, and is empty for a caller with no
       importing file.
    3. As a directory, for a language whose references may name one:
       every file of the language directly in it (`_files_in_directory`).
    """
    resolved = _substituted(path, resolution)
    # The index-file (package) form is not tried as a path suffix.
    found = _under_source_roots(resolved, repo_files, resolution) or _by_path_suffix(
        resolved, repo_files, resolution, importer_ext
    )
    if found is not None:
        return [found]
    if not resolution.package_directory:
        return []
    return _files_in_directory(resolved, repo_files, resolution.extensions)


def _climb(file_path: str, levels: int, module: str) -> str | None:
    """The path *module* names, read *levels* steps up from the file at *file_path*.

    One step is the file's own directory, two its parent.  None when the
    file sits too shallow for the climb, or nothing is left to name.
    """
    parts = PurePosixPath(file_path).parts
    if len(parts) < levels:
        return None
    segs = list(parts[:-levels])
    if "/" in module:
        segs.extend(s for s in module.split("/") if s)
    elif module:
        segs.append(module)
    if not segs:
        return None
    return "/".join(segs)


def _substituted(module_path: str, resolution: ImportResolution) -> str:
    """Apply the first matching prefix substitution, then the module style."""
    resolved = module_path
    for prefix, replacement in resolution.path_substitutions:
        if resolved.startswith(prefix):
            resolved = replacement + resolved[len(prefix) :]
            break
    if resolution.module_style is ModuleStyle.DOTTED and "." in resolved:
        resolved = resolved.replace(".", "/")
    return resolved


def _under_source_roots(
    resolved: str,
    repo_files: set[str],
    resolution: ImportResolution,
) -> str | None:
    """The file *resolved* names under a source root: as is, with an extension, or as an index file."""
    for root in resolution.source_roots:
        base = f"{root}/{resolved}" if root else resolved
        if base in repo_files:
            return base
        for ext in resolution.extensions:
            if f"{base}{ext}" in repo_files:
                return f"{base}{ext}"
        for idx in resolution.index_files:
            if f"{base}/{idx}" in repo_files:
                return f"{base}/{idx}"
    return None


def _by_path_suffix(
    resolved: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str,
) -> str | None:
    """The one file *resolved* names as a full-path suffix, anywhere in the repository.

    Needs at least two path segments, so a single-segment module (`os`,
    `io`) cannot match a nested `.../os.py`. Candidates are grouped by
    path without extension: more than one group means distinct files,
    which is ambiguity rather than a choice, and resolves to nothing.
    Files differing only by extension go to `_pick_by_importer`.
    """
    if "/" not in resolved:
        return None
    candidates = _suffix_matches(resolved, repo_files, resolution.extensions)
    if not candidates:
        return None
    groups: dict[str, list[str]] = {}
    for f in candidates:
        groups.setdefault(str(PurePosixPath(f).with_suffix("")), []).append(f)
    if len(groups) != 1:
        return None
    group = next(iter(groups.values()))
    if len(group) == 1:
        return group[0]
    return _pick_by_importer(group, importer_ext, resolution.own_extensions)


def _suffix_matches(
    name: str,
    repo_files: set[str],
    extensions: tuple[str, ...],
) -> list[str]:
    """Collect repo files matching *name* as a full-path suffix.

    A file matches when it equals `name{ext}` or ends with `/name{ext}` for
    some extension.
    """
    matches: list[str] = []
    for ext in extensions:
        target = f"{name}{ext}"
        suffix = f"/{target}"
        matches.extend(f for f in repo_files if f == target or f.endswith(suffix))
    return matches


def _pick_by_importer(
    candidates: list[str],
    importer_ext: str,
    own_extensions: frozenset[str],
) -> str | None:
    """Pick the candidate closest to the importing file.

    Ranks by: (0) extension equals the importer's own extension, (1) extension
    belongs to the importer's language, (2) anything else. Returns the single
    best-ranked candidate, or `None` if two candidates tie at the top rank.
    """

    def rank(f: str) -> int:
        ext = PurePosixPath(f).suffix
        if importer_ext and ext == importer_ext:
            return 0
        if ext in own_extensions:
            return 1
        return 2

    ranked = sorted(candidates, key=rank)
    best = rank(ranked[0])
    if sum(1 for f in ranked if rank(f) == best) != 1:
        return None
    return ranked[0]


def _files_in_directory(
    path: str,
    repo_files: set[str],
    extensions: tuple[str, ...],
) -> list[str]:
    """The language's files sitting directly in the directory *path*.

    Some references name a directory rather than a file, and the files
    inside it are not named by any convention: a Go package is its
    directory and every `.go` file in it belongs to the package, and a
    Terraform module is a directory of `.tf` files. Where a language does
    name one — `__init__.py`, `index.js`, `mod.rs` — `index_files`
    resolves it to that single file first.
    """
    if not path or not extensions:
        return []
    inside = f"{path}/"
    return sorted(
        f
        for f in repo_files
        if f.startswith(inside) and "/" not in f[len(inside) :] and f.endswith(extensions)
    )

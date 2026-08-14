"""Whether each language's own import syntax reaches an edge.

The scenarios in `cases_edges.py` hand the inference functions their
metadata, which tests resolution given a module path. These start from
source: they extract two real files and assert that the language's
extractor produces metadata the resolver can follow, which is what
`find-refs` depends on for that language.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.languages.edges import build_resolution_map, infer_import_edges
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager

from .cases_import_resolution import ImportScenario


@parametrize_with_cases("scenario", cases=".cases_import_resolution")
def test_language_import_reaches_the_imported_file(scenario: ImportScenario) -> None:
    """An import in one file infers an edge into the file it names."""
    manager = get_manager()
    chunks = list(
        extract_file(
            FileEntry(scenario.importer, "sha_importer", scenario.importer_source.encode()),
            scenario.language,
        )
    )
    # Each file is extracted as its own language, as a build does: a
    # component's script block imports a `.js` file, not another component.
    target_language = manager.detect_language(scenario.target) or scenario.language
    chunks += list(
        extract_file(
            FileEntry(scenario.target, "sha_target", scenario.target_source.encode()),
            target_language,
        )
    )

    edges = infer_import_edges(
        chunks,
        {scenario.importer, scenario.target},
        build_resolution_map(manager),
    )

    assert any(e.target_path == scenario.target for e in edges), (
        f"{scenario.language}: {scenario.importer} names {scenario.target}, "
        f"but inference produced {[e.target_path for e in edges]}"
    )

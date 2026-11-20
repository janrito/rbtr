"""Module resolution: which repository files an import's module names."""

from __future__ import annotations

import polars as pl
from pytest_cases import parametrize_with_cases

from rbtr.domain.edge_frames import EdgeChunkRow
from rbtr.domain.edges import requested_imports
from rbtr.domain.resolution import resolved_files
from rbtr.languages.resolution import language_resolutions

from .cases_resolution import ImportScenario


@parametrize_with_cases("scenario", cases=".cases_resolution")
def test_an_import_reaches_the_files_its_language_resolves(scenario: ImportScenario) -> None:
    resolutions = language_resolutions(scenario.registrations)
    rows = pl.DataFrame(
        {
            "id": ["import"],
            "file_path": [scenario.source_path],
            "line_start": [1],
            "kind": ["import"],
            "name": [scenario.module],
            "language": [scenario.language],
            "module": [scenario.module],
            "names": [scenario.names],
            "language_hint": [scenario.language_hint],
            "scope": [""],
            "bindings": [None],
            "refs": [None] * 1,
        }
    ).pipe(EdgeChunkRow.validate, cast=True)
    imports = requested_imports(rows, resolutions)
    found = resolved_files(imports, set(scenario.files), resolutions)["target_path"].to_list()
    assert sorted(found) == scenario.expected

"""Compile plugin declarations and supplied manifests into one `LanguageResolutionRow` per language."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import dataframely as dy
import polars as pl

from rbtr.domain.edge_frames import LanguageResolutionRow
from rbtr.languages.registration import LanguageRegistration


def reached_languages(
    registration: LanguageRegistration, by_id: Mapping[str, LanguageRegistration]
) -> list[LanguageRegistration]:
    """The languages *registration*'s imports may reach, itself first, then by id; those *by_id* holds."""
    targets = registration.import_targets or frozenset({registration.id})
    own = [registration.id] if registration.id in targets else []
    return [
        by_id[target] for target in own + sorted(targets - {registration.id}) if target in by_id
    ]


def language_resolutions(
    registrations: Sequence[LanguageRegistration], manifests: Mapping[str, str] | None = None
) -> dy.DataFrame[LanguageResolutionRow]:
    """Each of *registrations*' `LanguageResolutionRow`, manifest substitutions before declared ones.

    *manifests* maps a repository file's path to its text, for the files
    languages declare as their `manifest`. A row's `extensions` and
    `index_files` are those of the languages its imports may reach, in
    `reached_languages` order, each language's sorted; source roots and
    substitutions keep their declared order.
    """
    manifests = manifests or {}
    ordered = sorted(registrations, key=lambda registration: registration.id)
    by_id = {registration.id: registration for registration in ordered}
    reached = [reached_languages(registration, by_id) for registration in ordered]
    texts = [manifests.get(registration.manifest) for registration in ordered]
    return pl.DataFrame(
        {
            "language": [registration.id for registration in ordered],
            "extensions": [
                list(dict.fromkeys(e for target in targets for e in sorted(target.extensions)))
                for targets in reached
            ],
            "index_files": [
                list(dict.fromkeys(f for target in targets for f in sorted(target.index_files)))
                for targets in reached
            ],
            "source_roots": [list(registration.source_roots) for registration in ordered],
            "substitutions": [
                [
                    {"prefix": prefix, "replacement": replacement}
                    for prefix, replacement in (
                        registration.substitutions_from_manifest(text)
                        if registration.manifest and text is not None
                        else ()
                    )
                    + registration.path_substitutions
                ]
                for registration, text in zip(ordered, texts, strict=True)
            ],
            "module_style": [registration.module_style.value for registration in ordered],
            "own_extensions": [sorted(registration.extensions) for registration in ordered],
            "package_directory": [registration.package_directory for registration in ordered],
            "prose": [registration.prose for registration in ordered],
            "section_names": [registration.section_names.value for registration in ordered],
        },
        schema=LanguageResolutionRow.create_empty().schema,
    ).pipe(LanguageResolutionRow.validate, cast=True)

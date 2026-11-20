"""Python's import forms, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `named-imports`: A named import reaches each name: a function, a class,
  variables, and a relative module's class.
- `bare-import`: A bare import reaches a module's definitions, not its
  imports or the code no query names.
- `suffix-in-a-monorepo`: An absolute import resolves across a
  `packages/*/src` layout by its full path suffix.
- `suffix-in-two-packages`: A full path suffix two packages share is
  ambiguous, so it reaches neither.
- `imports-that-reach-nothing`: Imports whose module the repository does not
  hold reach nothing.
- `import-with-no-module`: An import its extractor records nothing for
  reaches files its words name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        pytest.param(
            "named-imports",
            {
                ("pkg/models.py", "User"): {("pkg/a.py", "from .models import User", "imports")},
                ("pkg/b.py", "Config"): {
                    ("pkg/a.py", "from pkg.b import helper, Config", "imports")
                },
                ("pkg/b.py", "helper"): {
                    ("pkg/a.py", "from pkg.b import helper, Config", "imports")
                },
                ("pkg/config.py", "settings"): {
                    ("pkg/a.py", "from pkg.config import settings, high", "imports")
                },
                ("pkg/config.py", "high"): {
                    ("pkg/a.py", "from pkg.config import settings, high", "imports")
                },
            },
            id="named-imports",
        ),
        pytest.param(
            "bare-import",
            {
                ("pkg/models.py", "User"): {
                    ("pkg/a.py", "import pkg.models", "imports"),
                    ("pkg/models.py", "load", "calls"),
                },
                ("pkg/models.py", "load"): {
                    ("pkg/a.py", "import pkg.models", "imports"),
                    ("pkg/models.py", "", "calls"),
                },
                ("pkg/models.py", "import os"): set(),
            },
            id="bare-import",
        ),
        pytest.param(
            "suffix-in-a-monorepo",
            {
                ("packages/core/src/core/helpers.py", "assist"): {
                    (
                        "packages/app/src/app/main.py",
                        "from core.helpers import assist",
                        "imports",
                    )
                },
                ("packages/core/src/core/models.py", "Model"): {
                    ("packages/app/src/app/main.py", "from core.models import Model", "imports")
                },
            },
            id="suffix-in-a-monorepo",
        ),
        pytest.param(
            "suffix-in-two-packages",
            {
                ("packages/a/src/core/models.py", "Model"): set(),
                ("packages/b/src/core/models.py", "Model"): set(),
            },
            id="suffix-in-two-packages",
        ),
        pytest.param(
            "imports-that-reach-nothing",
            {
                ("deep/nested/helpers.py", "assist"): set(),
                ("pkg/models.py", "User"): set(),
                ("pkg/utils.py", "util"): set(),
            },
            id="imports-that-reach-nothing",
        ),
        pytest.param(
            "import-with-no-module",
            {
                ("pkg/annotations.py", "note"): {
                    ("pkg/a.py", "from __future__ import annotations", "imports")
                },
                ("pkg/annotations.py", "other"): set(),
            },
            id="import-with-no-module",
        ),
    ],
)
def test_an_import_reaches_what_its_form_resolves_to(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "imports" / sample) as index:
        assert_reached(index, reached)

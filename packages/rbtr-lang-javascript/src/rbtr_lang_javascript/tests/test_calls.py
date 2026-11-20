"""What JavaScript's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `default-export`: a default import binds the module's default export,
  whatever name the importing file gives it.
- `member-order`: which inherited member a call on `this` reaches. A class
  has one base, so an override hides what it overrides; TypeScript's
  `implements` names an interface, whose signatures no call reaches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `import Button from "./widget.js"` binds `Widget`, the default export; `Button()` calls it.
        pytest.param(
            "default-export",
            {
                ("widget.js", "Widget"): {("app.js", "run", "calls")},
            },
            id="a-default-import-reaches-the-default-export",
            marks=pytest.mark.xfail(
                strict=True,
                reason="names only: a default import binds its local name, and the module "
                "defines nothing by that name, so the default export is not reached",
            ),
        ),
        # `Special extends Order extends Base`, `Order.save` overriding: `this.save()` is `Order.save`.
        pytest.param(
            "member-order",
            {
                ("override-js/base.js", "check"): {
                    ("override-js/base.js", "save", "calls"),
                    ("override-js/special.js", "run", "calls"),
                },
                ("override-js/base.js", "save"): {("override-js/order.js", "save", "calls")},
                ("override-js/order.js", "save"): {("override-js/special.js", "run", "calls")},
            },
            id="an-override-hides-its-bases-member-js",
        ),
        # `Special extends Order extends Base`, `Order.save` overriding: `this.save()` is `Order.save`.
        pytest.param(
            "member-order",
            {
                ("override-ts/base.ts", "check"): {
                    ("override-ts/base.ts", "save", "calls"),
                    ("override-ts/special.ts", "run", "calls"),
                },
                ("override-ts/base.ts", "save"): {("override-ts/order.ts", "save", "calls")},
                ("override-ts/order.ts", "save"): {("override-ts/special.ts", "run", "calls")},
            },
            id="an-override-hides-its-bases-member-ts",
        ),
        # `Order extends Base implements Saver`: `this.save()` is `Base.save`, not `Saver`'s signature.
        pytest.param(
            "member-order",
            {
                ("implements-ts/base.ts", "save"): {("implements-ts/order.ts", "run", "calls")},
                ("implements-ts/saver.ts", "save"): set(),
            },
            id="an-implemented-interfaces-signature-is-not-reached",
        ),
        # `const { log } = require("./util")` then `log()` calls `util.js`'s `log`.
        pytest.param(
            "commonjs",
            {
                ("util.js", "log"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", 'const { log } = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                    ("index.js", 'export { log } from "./util.js";', "imports"),
                    ("start.js", "start", "calls"),
                },
            },
            id="a-required-name-reaches-its-definition",
        ),
        # `export { log } from "./util.js"` in `index.js`: `log()` imported from it reaches `util.js`'s `log`.
        pytest.param(
            "commonjs",
            {
                ("util.js", "log"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", 'const { log } = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                    ("index.js", 'export { log } from "./util.js";', "imports"),
                    ("start.js", "start", "calls"),
                },
            },
            id="a-reexported-name-reaches-its-definition",
        ),
        # `const util = require("./util")` then `util.warn()` calls `util.js`'s `warn`.
        pytest.param(
            "commonjs",
            {
                ("util.js", "warn"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                },
            },
            id="a-required-module-reaches-its-members",
        ),
        # Each call reaches only its definition; a required name's variable is no target.
        #
        # A whole-module `require` imports every name, as `import * as` does.
        pytest.param(
            "commonjs",
            {
                ("util.js", "log"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", 'const { log } = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                    ("index.js", 'export { log } from "./util.js";', "imports"),
                    ("start.js", "start", "calls"),
                },
                ("util.js", "warn"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                },
            },
            id="commonjs-edges",
        ),
        # `import { log } from "./index.js"`, which re-exports it, imports `util.js`'s `log`.
        pytest.param(
            "commonjs",
            {
                ("util.js", "log"): {
                    ("app.js", 'const util = require("./util");', "imports"),
                    ("app.js", 'const { log } = require("./util");', "imports"),
                    ("app.js", "run", "calls"),
                    ("index.js", 'export { log } from "./util.js";', "imports"),
                    ("start.js", 'import { log } from "./index.js";', "imports"),
                    ("start.js", "start", "calls"),
                },
            },
            id="an-import-through-a-reexport-reaches-the-definition",
            marks=pytest.mark.xfail(
                strict=True,
                reason="an import edge reaches definitions in the module it names; index.js "
                "only re-exports log, so the import stops there though the call follows it",
            ),
        ),
    ],
)
def test_each_target_is_reached_by_exactly_the_chunks_named(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` of the sample under `samples/calls/` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "calls" / sample) as index:
        assert_reached(index, reached)

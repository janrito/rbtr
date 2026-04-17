"""Shared test data for index-family tests.

This file holds *only data* — no functions, no helpers, no
imports beyond what the constants need to exist.  Families that
need a different shape (e.g. GC's blob-sharing pairs) define
their own chunks locally instead of growing these to fit every
scenario.

Four chunk instances with deliberately distinct vocabulary so
FTS can distinguish them, and orthogonal unit-axis embedding
vectors so cosine similarity gives clean 0.0 / 1.0 values.

``MATH_FUNC`` and ``MATH_CLASS`` share ``blob_sha='blob_math'``
on purpose — two chunks in the same file, exercising the blob-
dedup path.
"""

from __future__ import annotations

from rbtr.index.models import Chunk, ChunkKind

MATH_FUNC = Chunk(
    id="math_1",
    blob_sha="blob_math",
    file_path="src/math_utils.py",
    kind=ChunkKind.FUNCTION,
    name="calculate_standard_deviation",
    content=(
        "def calculate_standard_deviation(values: list[float]) -> float:\n"
        "    mean = sum(values) / len(values)\n"
        "    variance = sum((x - mean) ** 2 for x in values) / len(values)\n"
        "    return variance ** 0.5\n"
    ),
    line_start=1,
    line_end=4,
)

HTTP_FUNC = Chunk(
    id="http_1",
    blob_sha="blob_http",
    file_path="src/api/client.py",
    kind=ChunkKind.FUNCTION,
    name="fetch_json_from_endpoint",
    content=(
        "async def fetch_json_from_endpoint(url: str, headers: dict) -> dict:\n"
        "    async with httpx.AsyncClient() as client:\n"
        "        response = await client.get(url, headers=headers)\n"
        "        response.raise_for_status()\n"
        "        return response.json()\n"
    ),
    line_start=10,
    line_end=15,
)

STRING_FUNC = Chunk(
    id="string_1",
    blob_sha="blob_string",
    file_path="src/text/normalize.py",
    kind=ChunkKind.FUNCTION,
    name="normalize_whitespace",
    content=(
        "def normalize_whitespace(text: str) -> str:\n"
        "    import re\n"
        "    collapsed = re.sub(r'\\s+', ' ', text)\n"
        "    return collapsed.strip()\n"
    ),
    line_start=1,
    line_end=4,
)

MATH_CLASS = Chunk(
    id="math_class_1",
    blob_sha="blob_math",  # shares blob with MATH_FUNC
    file_path="src/math_utils.py",
    kind=ChunkKind.CLASS,
    name="StatisticsCalculator",
    content=(
        "class StatisticsCalculator:\n"
        "    def __init__(self, data: list[float]):\n"
        "        self.data = data\n"
        "    def mean(self) -> float:\n"
        "        return sum(self.data) / len(self.data)\n"
    ),
    line_start=10,
    line_end=15,
)

ALL_CHUNKS = [MATH_FUNC, HTTP_FUNC, STRING_FUNC, MATH_CLASS]

# Orthogonal unit vectors → exact 0.0 / 1.0 cosine similarities.
VEC_MATH = [1.0, 0.0, 0.0]
VEC_HTTP = [0.0, 1.0, 0.0]
VEC_STRING = [0.0, 0.0, 1.0]

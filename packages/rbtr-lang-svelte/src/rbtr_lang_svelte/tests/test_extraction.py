"""SFC chunker edge cases: the markup template as a host chunk."""

from __future__ import annotations

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


def test_svelte_template_extracted_as_host_chunk(indexed: Indexed) -> None:
    """The SFC markup template is a searchable host (`svelte`) chunk.

    Distinct from the delegated `<script>`/`<style>` chunks: the template is
    the component's own markup, so it carries the host language.
    """
    src = """\
<script lang="ts">
  export let name: string = "world";
</script>

<h1 class="greeting">Hello {name}</h1>
"""
    with indexed({"input.svelte": src}) as build:
        chunks = outline(build)
    host = [c for c in chunks if c.language == "svelte"]
    # The script's tags are host chunks too (lines 1 and 3); the template is line 5.
    assert [(c.line_start, c.line_end) for c in host] == [(1, 1), (3, 3), (5, 5)]
    assert "greeting" in src.splitlines()[4]


def test_svelte_without_template_still_emits_host_chunks(indexed: Indexed) -> None:
    """A script-only SFC still has chunks in its own language: its block's tags.

    A chunk in the host language records the host language/version for
    dedup, so a svelte-plugin bump invalidates the file even with no
    template markup.
    """
    src = """\
<script lang="ts">
  export const x = 1;
</script>
"""
    with indexed({"input.svelte": src}) as build:
        chunks = outline(build)
    host = [c.line_start for c in chunks if c.language == "svelte"]
    assert [src.splitlines()[line - 1] for line in host] == ['<script lang="ts">', "</script>"]


def test_sfc_top_level_comment_is_its_own_chunk(indexed: Indexed) -> None:
    """A comment above the component's markup is searchable content.

    The template chunk excludes it, so it needs a chunk of its own or the
    banner explaining what a component is for cannot be found.
    """
    src = """\
<!--
  Greeting: shouts a name.
-->

<script>
  let name = "world";
</script>

<h1>{name}</h1>
"""
    with indexed({"input.svelte": src}) as build:
        chunks = outline(build)
    comments = [c for c in chunks if c.kind == ChunkKind.COMMENT]
    assert [(c.line_start, c.line_end) for c in comments] == [(1, 3)]

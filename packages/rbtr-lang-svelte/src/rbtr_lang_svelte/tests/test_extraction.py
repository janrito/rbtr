"""SFC chunker edge cases: the markup template as a host chunk."""

from __future__ import annotations

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


def test_svelte_template_extracted_as_host_chunk() -> None:
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
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "svelte")
    host = [c for c in chunks if c.language == "svelte"]
    assert host, "expected a svelte-language chunk for the template markup"
    assert "greeting" in host[0].content


def test_svelte_without_template_still_emits_host_chunk() -> None:
    """A script-only SFC still emits one (empty) host chunk.

    The host chunk records the host language/version for dedup, so a
    svelte-plugin bump invalidates the file even with no template markup.
    """
    src = '<script lang="ts">\n  export const x = 1;\n</script>\n'
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "svelte")
    host = [c for c in chunks if c.language == "svelte"]
    assert len(host) == 1
    assert host[0].content == ""

# Demo 1: Search & Retrieval
#
# Three ways to find code: by name, by snippet, by meaning.
# Uses rbtr on its own codebase.
#
# Record: vhs demo/index-search.tape

Output demo/output/index-search.gif

Set Shell "bash"
Set FontFamily "Iosevka"
Set FontSize 15
Set Width 1000
Set Height 550
Set Padding 20
Set Theme "Ayu Mirage"
Set TypingSpeed 40ms
Set CursorBlink false
Set PlaybackSpeed 2.0

Type "cd ~/apps/rbtr"
Enter
Wait

# ── Index ────────────────────────────────────────────────────

Type "# Index the repo"
Enter
Sleep 500ms

Type "rbtr index"
Enter
Wait

# ── Identifier search ────────────────────────────────────────

Type "# I know a name"
Enter
Sleep 500ms

Type 'rbtr search "Embedder" --limit 3'
Enter
Wait
Sleep 2.5s

# ── Code search ──────────────────────────────────────────────

Type "# I have a snippet from a traceback"
Enter
Sleep 500ms

Type 'rbtr search "model.embed(texts)" --limit 3'
Enter
Wait
Sleep 2.5s

# ── Concept search ───────────────────────────────────────────

Type "# I have a question"
Enter
Sleep 500ms

Type 'rbtr search "where is the embedding model loaded" --limit 3'
Enter
Wait
Sleep 3s

# ── Read symbol ──────────────────────────────────────────────

Type "# Show me the code"
Enter
Sleep 500ms

Type "rbtr read-symbol Embedder"
Enter
Wait
Sleep 2s

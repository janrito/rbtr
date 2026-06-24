# Demo 2: Structural Navigation
#
# Understand a codebase without reading it: file outlines,
# dependency graph, structural diffs.
#
# Before recording:
#   rbtr index main
#
# Record: vhs demo/structural-nav.tape

Output demo/output/structural-nav.gif

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

# ── File outline ─────────────────────────────────────────────

Type "# What's in this file?"
Enter
Sleep 500ms

Type "rbtr list-symbols packages/rbtr/src/rbtr/index/search.py"
Enter
Wait
Sleep 1s
ScrollUp 25
Sleep 2s
ScrollDown 25
Sleep 1s


# ── Read symbol ──────────────────────────────────────────────

Type "# Read one of them"
Enter
Sleep 500ms

Type "rbtr read-symbol fuse_scores"
Enter
Wait
Sleep 1s
ScrollUp 25
Sleep 3s
ScrollDown 25
Sleep 1s


# ── Find references ──────────────────────────────────────────

Type "# Who depends on the core data model?"
Enter
Sleep 500ms

Type "rbtr find-refs Chunk"
Enter
Wait
Sleep 1s
ScrollUp 25
Sleep 3s
ScrollDown 25
Sleep 1s

# ── Structural diff ──────────────────────────────────────────

Type "# What changed on this branch?"
Enter
Sleep 500ms

Type "rbtr changed-symbols v2026.6.0 main --file-path packages/rbtr/src/rbtr/index/search.py"
Enter
Wait
Sleep 2.5s

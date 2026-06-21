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

Type "cd ~/apps/rbtr"
Enter
Sleep 1s

# ── File outline ─────────────────────────────────────────────

Type "# What's in this file?"
Enter
Sleep 500ms

Type "rbtr list-symbols packages/rbtr/src/rbtr/index/search.py"
Enter
Sleep 5s

# ── Read symbol ──────────────────────────────────────────────

Type "# Read one of them"
Enter
Sleep 500ms

Type "rbtr read-symbol fuse_scores"
Enter
Sleep 6s

# ── Find references ──────────────────────────────────────────

Type "# Who depends on the core data model?"
Enter
Sleep 500ms

Type "rbtr find-refs Chunk"
Enter
Sleep 5s

# Scroll up so the viewer can see the top of the list
ScrollUp 5
Sleep 3s
ScrollDown 5
Sleep 2s

# ── Structural diff ──────────────────────────────────────────

Type "# What changed on this branch?"
Enter
Sleep 500ms

Type "rbtr changed-symbols main HEAD --file-path packages/rbtr/src/rbtr/index/search.py"
Enter
Sleep 5s

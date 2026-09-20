# Demo 2: Structural Navigation
#
# Understand a module without reading it: outline, source,
# dependents, and what this branch changed.
#
# Before recording:
#   rbtr watch main          # changed-symbols needs both refs indexed
#   git checkout <a branch with commits on top of main>
#
# Record: vhs demo/structural-nav.tape

Output demo/output/structural-nav.gif

Set Shell "bash"
Set FontFamily "Iosevka"
Set FontSize 15
Set Width 1000
Set Height 750
Set Padding 20
Set Theme "Ayu Mirage"
Set TypingSpeed 40ms
Set CursorBlink false
Set PlaybackSpeed 2.0


Type "cd ~/apps/rbtr"
Enter
Wait@60s

# ── File outline ─────────────────────────────────────────────

Type "# What's in this file?"
Enter
Sleep 500ms

Type "rbtr list-symbols packages/rbtr/src/rbtr/index/watch.py"
Enter
Wait@60s
Sleep 3s


# ── Read symbol ──────────────────────────────────────────────

Type "# Read one of them, by name alone"
Enter
Sleep 500ms

Type "rbtr read-symbol remove_stale_refs"
Enter
Wait@60s
Sleep 4s


# ── Find references ──────────────────────────────────────────

Type "# Who depends on this type?"
Enter
Sleep 500ms

Type "rbtr find-refs Scope"
Enter
Wait@60s
Sleep 3s


# ── Structural diff ──────────────────────────────────────────

Type "# And what did this branch change here?"
Enter
Sleep 500ms

Type "rbtr changed-symbols main HEAD --file-path packages/rbtr/src/rbtr/index/watch.py"
Enter
Wait@60s
Sleep 3s

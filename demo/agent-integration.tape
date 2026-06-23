# Demo 3: Agent Integration
#
# The LLM chains rbtr tools to navigate code — no shell
# commands, no file paths.
#
# LLM output varies. Record 2-3 takes, keep the best.
#
# Record: vhs demo/agent-integration.tape

Output demo/output/agent-integration.gif

Set Shell "bash"
Set FontFamily "Iosevka"
Set FontSize 14
Set Width 1100
Set Height 650
Set Padding 20
Set Theme "Ayu Mirage"
Set TypingSpeed 40ms
Set CursorBlink false

Type "cd ~/apps/rbtr"
Enter
Sleep 1s

# ── Start pi ─────────────────────────────────────────────────

Type "# The footer shows the index is live"
Enter
Sleep 500ms

Type "pi --model openrouter/z-ai/glm-5.2 --no-skills --no-context-files --no-prompt-templates --no-themes --no-extensions -e packages/pi-rbtr"
Enter
Sleep 8s

# ── Concept question ─────────────────────────────────────────

Type "How does search fusion work in this codebase?"
Enter
Sleep 30s

# ── Branch changes ───────────────────────────────────────────

Type "What changed on the current branch vs main?"
Enter
Sleep 25s

# Let the final answer settle on screen
Sleep 10s

# ── Exit ─────────────────────────────────────────────────────

Ctrl+C
Sleep 1s

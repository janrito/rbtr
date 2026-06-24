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
Set PlaybackSpeed 2.0

Type "cd ~/apps/rbtr"
Enter
Sleep 1s

# ── Start pi ─────────────────────────────────────────────────

Type "# The footer shows the index is live"
Enter
Sleep 500ms

Type@20ms "pi --model openrouter/z-ai/glm-5.2 --no-skills --no-context-files --no-prompt-templates --no-themes --no-extensions -e packages/pi-rbtr"
Enter
Sleep 5s

# ── Concept question ─────────────────────────────────────────

Type "How does search fusion work?"
Enter
Sleep 60s

# ── Exit ─────────────────────────────────────────────────────

Ctrl+C
Sleep 1s

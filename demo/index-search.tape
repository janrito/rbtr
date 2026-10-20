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
Set Height 650
Set Padding 20
Set Theme "Ayu Mirage"
Set TypingSpeed 40ms
Set CursorBlink false
Set PlaybackSpeed 2.0

Type "cd ~/apps/rbtr"
Enter
Wait@120s

Type "# Keep this repo indexed"
Enter
Sleep 500ms

Type "rbtr watch"
Enter
Wait@300s
Sleep 1.5s

Type "# I know a name"
Enter
Sleep 500ms

Type 'rbtr search "Embedder" --limit 3'
Enter
Wait@120s
Sleep 2.5s

Type "# I have a snippet from a traceback"
Enter
Sleep 500ms

Type 'rbtr search "model.embed(texts)" --limit 3'
Enter
Wait@120s
Sleep 2.5s

Type "# I have a question"
Enter
Sleep 500ms

Type 'rbtr search "where is the embedding model loaded" --limit 3'
Enter
Wait@120s
Sleep 3s

Type "# Show me the code"
Enter
Sleep 500ms

Type "rbtr read-symbol Embedder"
Enter
Wait@120s
Sleep 2s

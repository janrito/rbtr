# rbtr-lang-bash

Bash / shell support for [rbtr](../rbtr). **Default** — installed with
core `rbtr`, no extra required.

## What it ingests

- **Functions** — both `name() { ... }` and `function name { ... }`
  forms. All functions are top-level (bash has no nesting).
- **Top-level variable assignments** — `MAX=100`, and `alias` names.
- **Imports** — `source file` and `. file` commands.

Bash has no classes, methods, or module structure, so none are emitted.

## Chunks produced

`scope` is always empty. `name` is the function/variable identifier.

```bash
deploy() { echo deploying; }   # function "deploy"
function setup { ...; }         # function "setup"
MAX=100                         # variable "MAX"
alias ll="ls -l"                # variable "ll"  (the fused `=` is stripped)
source ./env.sh                 # import  "./env.sh"
. /etc/profile                  # import  "/etc/profile"
```

A leading `#` comment run above a function is attached as its
documentation; a `#!/bin/bash` shebang attaches to the first function
only when no blank line separates them.

## Embedded / injected chunks

None. Bash does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-bash` grammar. No dependency on other language
plugins.

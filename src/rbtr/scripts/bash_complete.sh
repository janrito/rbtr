#!/usr/bin/env bash
# bash_complete.sh — query bash's programmable completion for a command.
#
# Usage:
#   bash bash_complete.sh <command> <cword> <cmdline> <word0> [word1 ...]
#
# Arguments:
#   $1  command name (e.g. "git")
#   $2  COMP_CWORD — zero-based index of the word being completed
#   $3  full command line as a single string
#   $4… COMP_WORDS array elements
#
# The script:
#   1. Sources the bash-completion framework (searches well-known paths
#      across macOS Homebrew, Linux distros, NixOS, MacPorts, snap).
#   2. If the framework didn't register a completion function for the
#      command, manually searches completion directories for a matching
#      script and sources it.
#   3. Invokes the registered completion function with the standard
#      COMP_LINE / COMP_POINT / COMP_WORDS / COMP_CWORD variables set.
#   4. Prints each COMPREPLY entry on its own line to stdout.
#
# Exit codes:
#   0  always (empty output means no completions available)

# NOTE: no `set -e` — completion functions routinely use commands that
# return non-zero as part of normal operation (e.g. `complete -p` when
# no completion is registered yet).

cmd="$1"
shift
cword="$1"
shift
cmd_line="$1"
shift
# remaining args are COMP_WORDS

# ── Source bash-completion framework if available ─────────────────────
for f in \
    /opt/homebrew/share/bash-completion/bash_completion \
    /usr/share/bash-completion/bash_completion \
    /usr/local/share/bash-completion/bash_completion \
    /etc/bash_completion \
    /opt/local/share/bash-completion/bash_completion \
    /run/current-system/sw/share/bash-completion/bash_completion; do
    [ -f "$f" ] && {
        # shellcheck source=/dev/null
        source "$f" 2>/dev/null
        break
    }
done

# ── If framework didn't register completion, find the file manually ──
if ! complete -p "$cmd" &>/dev/null; then
    for dir in \
        "${BASH_COMPLETION_USER_DIR:-$HOME/.local/share/bash-completion}/completions" \
        /opt/homebrew/share/bash-completion/completions \
        /opt/homebrew/etc/bash_completion.d \
        /usr/local/share/bash-completion/completions \
        /usr/local/etc/bash_completion.d \
        /usr/share/bash-completion/completions \
        /etc/bash_completion.d \
        /run/current-system/sw/share/bash-completion/completions \
        /opt/local/share/bash-completion/completions \
        /opt/local/etc/bash_completion.d \
        /snap/core/current/usr/share/bash-completion/completions; do
        for file in "$dir/${cmd}-completion.bash" "$dir/${cmd}.bash" "$dir/$cmd"; do
            [ -f "$file" ] && {
                # shellcheck source=/dev/null
                source "$file" 2>/dev/null
                break 2
            }
        done
    done
fi

# ── Invoke the registered completion function ────────────────────────
func=$(complete -p "$cmd" 2>/dev/null | sed -n 's/.*-F \([^ ]*\).*/\1/p')
[ -z "$func" ] && exit 0

COMP_LINE="$cmd_line"
COMP_POINT=${#cmd_line}
COMP_WORDS=("$@")
COMP_CWORD=$cword

$func "${COMP_WORDS[0]}" "${COMP_WORDS[$cword]}" "${COMP_WORDS[$((cword > 0 ? cword - 1 : 0))]}" 2>/dev/null

printf '%s\n' "${COMPREPLY[@]}"

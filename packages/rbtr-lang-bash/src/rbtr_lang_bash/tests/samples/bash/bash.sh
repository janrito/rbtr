#!/usr/bin/env bash
# Greeter — print greetings for named recipients.
#
# Bash has no classes or methods; the plugin extracts functions,
# top-level variable assignments (including export/declare/readonly),
# aliases, and source/. imports.

source ./lib/colours.sh
. /etc/greeter.conf

DEFAULT_GREETING="Hello"
LOCALE="${LANG:-en}"
export API_URL="https://example.com"
readonly MAX_RETRIES=3
declare -i COUNTER=0
alias greet="format_greeting"

# Format a greeting for a single recipient.
format_greeting() {
    local name="$1"
    echo "${DEFAULT_GREETING}, ${name} (${LOCALE})"
}

# Greet every argument in turn.
greet_all() {
    for name in "$@"; do
        format_greeting "$name"
    done
}

greet_all "$@"

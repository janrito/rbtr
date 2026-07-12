// Package util holds greeting helpers.
package util

import "strings"

// Trim removes surrounding whitespace from a name.
func Trim(name string) string {
	return strings.TrimSpace(name)
}

// Package util holds greeting helpers.
//
// Imported as example.com/greeter/greeter: the path carries the module
// prefix go.mod declares, and this file is named for what it holds rather
// than for the directory that is the package.
package util

import "strings"

// Trim removes surrounding whitespace from a name.
func Trim(name string) string {
	return strings.TrimSpace(name)
}

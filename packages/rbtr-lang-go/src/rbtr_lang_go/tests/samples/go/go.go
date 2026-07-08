// Package greet formats greetings for named recipients.
//
// The Go plugin extracts functions, methods (functions with a
// receiver), type declarations (struct, interface, and alias, as
// classes), const/var declarations (as variables), and imports.
package greet

import (
	"fmt"

	"greeter/util"
)

// DefaultGreeting is the fallback prefix.
const DefaultGreeting = "Hello"

var fallbackLocale = "en"

// Temperature is an alias for the underlying float type.
type Temperature = float64

// Greeter holds a greeting prefix.
type Greeter struct {
	Prefix string
}

// Formatter renders a greeting for a name.
type Formatter interface {
	Format(name string) string
}

// Greet greets a single recipient.
func (g Greeter) Greet(name string) string {
	return fmt.Sprintf("%s, %s", g.Prefix, util.Trim(name))
}

// FormatGreeting formats a greeting via a Greeter.
func FormatGreeting(g Greeter, name string) string {
	return g.Greet(name)
}

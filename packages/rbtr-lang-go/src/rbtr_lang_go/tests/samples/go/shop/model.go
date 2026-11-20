// Package shop holds orders; its files share their names without imports.
package shop

// Model is embedded in Order, which gets its methods.
type Model struct{}

// Check reports whether a model is valid.
func (m Model) Check() bool {
	return valid()
}

func valid() bool {
	return true
}

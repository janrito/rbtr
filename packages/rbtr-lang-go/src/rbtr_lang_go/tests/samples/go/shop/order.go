package shop

// Order embeds Model.
type Order struct {
	Model
	Name string
}

// NewOrder builds an order with a function its package's other file defines.
func NewOrder(name string) Order {
	if !valid() {
		return Order{}
	}
	return Order{Name: name}
}

package shop

type Order struct{}

func (o *Order) check() bool {
	return true
}

func (o *Order) Save() bool {
	return o.check()
}

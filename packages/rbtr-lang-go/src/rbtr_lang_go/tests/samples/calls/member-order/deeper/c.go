package deeper

type C struct {
	A
	B
}

func (c *C) Run() string {
	return c.M()
}

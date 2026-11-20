package override

type Base struct{}

func (b *Base) Check() bool {
	return true
}

func (b *Base) Save() bool {
	return b.Check()
}

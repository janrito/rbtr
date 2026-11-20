package diamond

type Bottom struct {
	Left
	Right
}

func (b *Bottom) Run() string {
	return b.M()
}

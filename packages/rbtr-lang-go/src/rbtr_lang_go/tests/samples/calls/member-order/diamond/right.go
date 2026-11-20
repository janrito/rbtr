package diamond

type Right struct {
	Top
}

func (r *Right) M() string {
	return "right"
}

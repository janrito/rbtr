package override

type Special struct {
	Order
}

func (s *Special) Run() bool {
	return s.Save() && s.Check()
}

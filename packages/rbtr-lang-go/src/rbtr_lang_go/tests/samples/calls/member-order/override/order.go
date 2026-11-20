package override

type Order struct {
	Base
}

func (o *Order) Save() bool {
	return o.Base.Save()
}

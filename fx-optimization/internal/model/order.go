package model

type Side int

const (
	Buy Side = iota
	Sell
)

func (s Side) String() string {
	switch s {
	case Buy:
		return "BUY"
	case Sell:
		return "SELL"
	default:
		return "UNKNOWN"
	}
}

type Order struct {
	Pair       string
	Side       Side
	Notional   float64
	HorizonSec int
	Slices     int
}

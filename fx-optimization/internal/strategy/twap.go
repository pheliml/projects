package strategy

import "fxopt/internal/model"

// TWAP splits the order into equal-sized slices distributed uniformly over time.
func TWAP(order model.Order) []float64 {
	sliceQty := order.Notional / float64(order.Slices)
	schedule := make([]float64, order.Slices)
	for i := range schedule {
		schedule[i] = sliceQty
	}
	return schedule
}

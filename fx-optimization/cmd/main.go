package main

import (
	"fmt"
	"math/rand"
	"time"

	"fxopt/internal/cost"
	"fxopt/internal/model"
	"fxopt/internal/sim"
	"fxopt/internal/strategy"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	order := model.Order{
		Pair:       "EUR/USD",
		Side:       model.buy,
		Notional:   10_000_000,
		HorizonSec: 300,
		Slices:     10,
	}

	market := sim.GenerateMarket(order.Slices)

	schedule := strategy.TWAP(order)

	totalCost := cost.ExecutionCost(order, schedule, market)

	fmt.Println("------------------------")
	fmt.Printf("Pair: %s\n", order.Pair)
	fmt.Printf("Side: %s\n", order.Side)
	fmt.Printf("Notional: %.0f\n", order.Notional)
	fmt.Printf("Slices: %d\n\n", order.Slices)

	fmt.Printf("Execution schedule:\n")
	for i, q := range schedule {
		fmt.Printf("  Slice %d: %.0f\n", i+1, q)
	}

	fmt.Printf("\nTotal execution cost: %.2f USD\n", totalCost)
}

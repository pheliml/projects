package main

import (
	"log/slog"
	"math/rand"
	"os"
	"sync"
	"time"

	"fxopt/internal/cost"
	"fxopt/internal/model"
	"fxopt/internal/sim"
	"fxopt/internal/stats"
	"fxopt/internal/strategy"
)

// run executes a single simulation and returns the total execution cost in the
// order's quote currency.
func run(id int, seed int64, logger *slog.Logger) float64 {
	rng := rand.New(rand.NewSource(seed))

	order := model.Order{
		Pair:       "EUR/USD",
		Notional:   10_000_000,
		HorizonSec: 300,
		Slices:     10000,
	}

	market := sim.GenerateMarket(order.Slices, rng)
	schedule := strategy.TWAP(order)
	totalCost := cost.ExecutionCost(order, schedule, market)

	logger.Debug("Run complete", "run", id, "cost", totalCost, "cur", "USD")

	return totalCost
}

func main() {

	const numRuns = 100

	start := time.Now()
	defer func() {
		slog.Info("Program execution time", "duration", time.Since(start))
	}()

	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	logger := slog.New(handler)
	slog.SetDefault(logger)

	// Each goroutine writes only its own index, so the WaitGroup is the only
	// synchronisation needed to publish the results safely.
	costs := make([]float64, numRuns)

	var wg sync.WaitGroup
	for i := range numRuns {
		// Draw seeds from the global source, which is randomly seeded and safe for
		// concurrent use. Seeding per run from time.Now() would hand many of these
		// near-simultaneous goroutines the same stream, collapsing the variance we
		// are trying to measure.
		seed := rand.Int63()

		wg.Add(1)
		go func() {
			defer wg.Done()
			costs[i] = run(i, seed, logger)
		}()
	}
	wg.Wait()

	logger.Info("Execution cost summary", "cur", "USD", "cost", stats.Summarise(costs))
}

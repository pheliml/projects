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

func run(id int, seed int64, logger *slog.Logger) (cost.Breakdown, error) {
	rng := rand.New(rand.NewSource(seed))

	order := model.Order{
		Pair:       "EUR/USD",
		Side:       model.Buy,
		Notional:   10_000_000,
		HorizonSec: 300,
		Slices:     10000,
	}

	market := sim.GenerateMarket(sim.EURUSD(), order.Slices, order.Horizon(), rng)
	schedule := strategy.TWAP(order)

	breakdown, err := cost.Shortfall(cost.EURUSDParams(), order, schedule, market)
	if err != nil {
		return cost.Breakdown{}, err
	}

	logger.Debug("Run complete", "run", id, "cost", breakdown.Total, "bps", breakdown.Bps(), "cur", "USD")

	return breakdown, nil
}

func main() {

	const numRuns = 10000

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
	results := make([]cost.Breakdown, numRuns)
	errs := make([]error, numRuns)

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
			results[i], errs[i] = run(i, seed, logger)
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			logger.Error("Run failed", "run", i, "err", err)
			os.Exit(1)
		}
	}

	logger.Info("Implementation shortfall", "cur", "USD",
		"total", stats.Summarise(project(results, func(b cost.Breakdown) float64 { return b.Total })))
	logger.Info("Shortfall in basis points",
		"bps", stats.Summarise(project(results, cost.Breakdown.Bps)))
	logger.Info("Cost components", "cur", "USD",
		"spread", stats.Summarise(project(results, func(b cost.Breakdown) float64 { return b.Spread })),
		"temporary", stats.Summarise(project(results, func(b cost.Breakdown) float64 { return b.Temporary })),
		"permanent", stats.Summarise(project(results, func(b cost.Breakdown) float64 { return b.Permanent })),
		"timing", stats.Summarise(project(results, func(b cost.Breakdown) float64 { return b.Timing })))
}

func project(results []cost.Breakdown, field func(cost.Breakdown) float64) []float64 {
	values := make([]float64, len(results))
	for i, r := range results {
		values[i] = field(r)
	}
	return values
}

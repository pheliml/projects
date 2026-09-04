package cost

import (
	"math"
	"math/rand"
	"testing"

	"fxopt/internal/model"
	"fxopt/internal/sim"
	"fxopt/internal/strategy"
)

func testOrder(side model.Side, slices int) model.Order {
	return model.Order{Pair: "EUR/USD", Side: side, Notional: 10e6, HorizonSec: 300, Slices: slices}
}

func flatMarket(slices int) []model.MarketSlice {
	market := make([]model.MarketSlice, slices)
	for i := range market {
		market[i] = model.MarketSlice{MidPrice: 1.0850, Spread: 0.0001, Liquidity: 6e6, Volatility: 0.07}
	}
	return market
}

func TestComponentsSumToShortfall(t *testing.T) {
	for _, side := range []model.Side{model.Buy, model.Sell} {
		order := testOrder(side, 500)
		rng := rand.New(rand.NewSource(3))
		market := sim.GenerateMarket(sim.EURUSD(), order.Slices, order.Horizon(), rng)

		b, err := Shortfall(EURUSDParams(), order, strategy.TWAP(order), market)
		if err != nil {
			t.Fatalf("Shortfall: %v", err)
		}

		dir := 1.0
		if side == model.Sell {
			dir = -1.0
		}
		want := dir * (b.AvgPrice - b.Arrival) * b.Quantity
		if math.Abs(b.Total-want) > 1e-6*math.Abs(want) {
			t.Errorf("%s: Total = %.6f, want %.6f from (AvgPrice-Arrival)*Quantity", side, b.Total, want)
		}
		if sum := b.Spread + b.Temporary + b.Permanent + b.Timing; math.Abs(b.Total-sum) > 1e-9 {
			t.Errorf("%s: components sum to %.9f, Total = %.9f", side, sum, b.Total)
		}
	}
}

func TestSideSymmetry(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	buy, sell := testOrder(model.Buy, 500), testOrder(model.Sell, 500)
	market := sim.GenerateMarket(sim.EURUSD(), buy.Slices, buy.Horizon(), rng)

	b, err := Shortfall(EURUSDParams(), buy, strategy.TWAP(buy), market)
	if err != nil {
		t.Fatalf("Shortfall(buy): %v", err)
	}
	s, err := Shortfall(EURUSDParams(), sell, strategy.TWAP(sell), market)
	if err != nil {
		t.Fatalf("Shortfall(sell): %v", err)
	}

	for _, c := range []struct {
		name      string
		buy, sell float64
	}{
		{"Spread", b.Spread, s.Spread},
		{"Temporary", b.Temporary, s.Temporary},
		{"Permanent", b.Permanent, s.Permanent},
		{"Timing", b.Timing, -s.Timing},
	} {
		if math.Abs(c.buy-c.sell) > 1e-9 {
			t.Errorf("%s: buy = %.9f, sell-equivalent = %.9f", c.name, c.buy, c.sell)
		}
	}
}

func TestImpactIsGridInvariant(t *testing.T) {
	base := 0.0
	for _, slices := range []int{50, 500, 5000} {
		order := testOrder(model.Buy, slices)
		b, err := Shortfall(EURUSDParams(), order, strategy.TWAP(order), flatMarket(slices))
		if err != nil {
			t.Fatalf("Shortfall: %v", err)
		}

		impact := b.Temporary + b.Permanent
		if base == 0 {
			base = impact
			continue
		}
		if math.Abs(impact-base) > 0.01*base {
			t.Errorf("slices=%d: impact = %.4f, want within 1%% of %.4f", slices, impact, base)
		}
	}
}

func TestPermanentImpactFollowsSquareRootLaw(t *testing.T) {
	const slices = 500
	measure := func(notional float64) float64 {
		order := testOrder(model.Buy, slices)
		order.Notional = notional
		b, err := Shortfall(EURUSDParams(), order, strategy.TWAP(order), flatMarket(slices))
		if err != nil {
			t.Fatalf("Shortfall: %v", err)
		}
		return b.Permanent
	}

	small, large := measure(10e6), measure(40e6)
	if got := large / small; math.Abs(got-8) > 0.01 {
		t.Errorf("quadrupling notional scaled permanent impact by %.4f, want 8 (Q^1.5)", got)
	}
}

func TestConcentrationCostsMoreThanSpreading(t *testing.T) {
	const slices = 500
	order := testOrder(model.Buy, slices)
	market := flatMarket(slices)

	even, err := Shortfall(EURUSDParams(), order, strategy.TWAP(order), market)
	if err != nil {
		t.Fatalf("Shortfall(even): %v", err)
	}

	burst := make([]float64, slices)
	for i := range slices / 4 {
		burst[i] = order.Notional / float64(slices/4)
	}
	fast, err := Shortfall(EURUSDParams(), order, burst, market)
	if err != nil {
		t.Fatalf("Shortfall(burst): %v", err)
	}

	if fast.Temporary <= even.Temporary {
		t.Errorf("temporary impact for a burst = %.4f, want more than the %.4f for an even schedule",
			fast.Temporary, even.Temporary)
	}
}

func TestTimingRiskDominatesDispersion(t *testing.T) {
	const runs = 500
	order := testOrder(model.Buy, 500)

	var sum, sumSq, spreadSum float64
	for r := range runs {
		rng := rand.New(rand.NewSource(int64(r) + 1))
		market := sim.GenerateMarket(sim.EURUSD(), order.Slices, order.Horizon(), rng)
		b, err := Shortfall(EURUSDParams(), order, strategy.TWAP(order), market)
		if err != nil {
			t.Fatalf("Shortfall: %v", err)
		}
		sum += b.Timing
		sumSq += b.Timing * b.Timing
		spreadSum += b.Spread
	}

	mean := sum / runs
	sd := math.Sqrt(sumSq/runs - mean*mean)
	if stdErr := sd / math.Sqrt(runs); math.Abs(mean) > 3*stdErr {
		t.Errorf("mean timing cost = %.2f, more than 3 standard errors (%.2f) from zero", mean, stdErr)
	}
	if sd < spreadSum/runs {
		t.Errorf("timing standard deviation %.2f is below mean spread cost %.2f; "+
			"the price path is barely reaching the cost function", sd, spreadSum/runs)
	}
}

func TestShortfallRejectsBadInput(t *testing.T) {
	order := testOrder(model.Buy, 10)
	for _, c := range []struct {
		name     string
		schedule []float64
		market   []model.MarketSlice
	}{
		{"empty schedule", nil, flatMarket(10)},
		{"market shorter than schedule", make([]float64, 10), flatMarket(9)},
	} {
		if _, err := Shortfall(EURUSDParams(), order, c.schedule, c.market); err == nil {
			t.Errorf("%s: got nil error, want failure", c.name)
		}
	}
}

func TestBpsHandlesZeroNotional(t *testing.T) {
	if got := (Breakdown{Total: 100}).Bps(); got != 0 {
		t.Errorf("Bps() on a zero-notional breakdown = %v, want 0", got)
	}
}

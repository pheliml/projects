package sim

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

const horizon = 300 * time.Second

func TestGridInvariance(t *testing.T) {
	p := EURUSD()
	const runs = 400

	type moments struct{ vol, spread, liq, terminalVar float64 }
	measure := func(slices int) moments {
		var m moments
		for r := range runs {
			rng := rand.New(rand.NewSource(int64(r) + 1))
			market := GenerateMarket(p, slices, horizon, rng)
			for _, ms := range market {
				m.vol += ms.Volatility
				m.spread += ms.Spread
				m.liq += ms.Liquidity
			}
			ret := market[len(market)-1].MidPrice/p.BaseMid - 1
			m.terminalVar += ret * ret
		}
		n := float64(runs * slices)
		return moments{m.vol / n, m.spread / n, m.liq / n, m.terminalVar / runs}
	}

	base := measure(200)
	for _, slices := range []int{50, 1000, 4000} {
		got := measure(slices)
		for _, c := range []struct {
			name      string
			got, want float64
			tolerance float64
		}{
			{"volatility", got.vol, base.vol, 0.05},
			{"spread", got.spread, base.spread, 0.05},
			{"liquidity", got.liq, base.liq, 0.05},
			{"terminal variance", got.terminalVar, base.terminalVar, 0.40},
		} {
			if relDiff(c.got, c.want) > c.tolerance {
				t.Errorf("slices=%d: %s = %g, want within %.0f%% of %g (200 slices)",
					slices, c.name, c.got, c.tolerance*100, c.want)
			}
		}
	}
}

func TestMidIsMartingale(t *testing.T) {
	p := EURUSD()
	const runs = 4000

	sum, sumSq := 0.0, 0.0
	for r := range runs {
		rng := rand.New(rand.NewSource(int64(r) + 1))
		market := GenerateMarket(p, 500, horizon, rng)
		ret := market[len(market)-1].MidPrice/p.BaseMid - 1
		sum += ret
		sumSq += ret * ret
	}

	mean := sum / runs
	stdErr := math.Sqrt(sumSq/runs-mean*mean) / math.Sqrt(runs)
	if math.Abs(mean) > 3*stdErr {
		t.Errorf("terminal return mean = %.3e, more than 3 standard errors (%.3e) from zero", mean, stdErr)
	}
}

func TestSpreadPersistsAcrossSlices(t *testing.T) {
	p := EURUSD()
	const runs = 300

	dispersion := func(slices int) float64 {
		sum, sumSq := 0.0, 0.0
		for r := range runs {
			rng := rand.New(rand.NewSource(int64(r) + 1))
			market := GenerateMarket(p, slices, horizon, rng)
			avg := 0.0
			for _, ms := range market {
				avg += ms.Spread
			}
			avg /= float64(slices)
			sum += avg
			sumSq += avg * avg
		}
		mean := sum / runs
		return math.Sqrt(sumSq/runs-mean*mean) / mean
	}

	coarse, fine := dispersion(100), dispersion(10000)
	if iid := coarse / 10; fine < iid*4 {
		t.Errorf("coefficient of variation fell from %.4f to %.4f over a 100x finer grid; "+
			"that is near the %.4f expected of independent draws, so the spread process is not persistent",
			coarse, fine, iid)
	}
}

func TestSpreadRespectsTickGrid(t *testing.T) {
	p := EURUSD()
	rng := rand.New(rand.NewSource(7))
	minSpread := p.MinSpreadTicks * p.TickSize

	for i, ms := range GenerateMarket(p, 5000, horizon, rng) {
		if ms.Spread < minSpread {
			t.Fatalf("slice %d: spread %g below the %g floor", i, ms.Spread, minSpread)
		}
		for _, v := range []struct {
			name string
			val  float64
		}{{"spread", ms.Spread}, {"mid", ms.MidPrice}} {
			if ticks := v.val / p.TickSize; math.Abs(ticks-math.Round(ticks)) > 1e-6 {
				t.Fatalf("slice %d: %s %.10f is not on the %g tick grid", i, v.name, v.val, p.TickSize)
			}
		}
	}
}

// Aggregated across seeds, not slices: one 300s path holds only a handful of
// independent regime draws.
func TestStressCouplesSpreadAndLiquidity(t *testing.T) {
	p := EURUSD()
	const runs = 300

	var calmSpread, calmLiq, stressSpread, stressLiq float64
	var calm, stress int
	for r := range runs {
		rng := rand.New(rand.NewSource(int64(r) + 1))
		for _, ms := range GenerateMarket(p, 2000, horizon, rng) {
			if ms.Volatility > p.AnnualVol {
				stressSpread += ms.Spread
				stressLiq += ms.Liquidity
				stress++
			} else {
				calmSpread += ms.Spread
				calmLiq += ms.Liquidity
				calm++
			}
		}
	}
	if calm == 0 || stress == 0 {
		t.Fatalf("degenerate split: %d calm, %d stressed slices", calm, stress)
	}

	if got, want := stressSpread/float64(stress), calmSpread/float64(calm); got <= want {
		t.Errorf("mean spread in high-volatility slices = %g, want wider than the %g in calm ones", got, want)
	}
	if got, want := stressLiq/float64(stress), calmLiq/float64(calm); got >= want {
		t.Errorf("mean liquidity in high-volatility slices = %g, want thinner than the %g in calm ones", got, want)
	}
}

func TestGenerateMarketRejectsEmptyGrid(t *testing.T) {
	if got := GenerateMarket(EURUSD(), 0, horizon, rand.New(rand.NewSource(1))); got != nil {
		t.Errorf("GenerateMarket(slices=0) = %v, want nil", got)
	}
}

func relDiff(a, b float64) float64 {
	if b == 0 {
		return math.Abs(a)
	}
	return math.Abs(a-b) / math.Abs(b)
}

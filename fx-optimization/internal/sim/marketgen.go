package sim

import (
	"math"
	"math/rand"
	"time"

	"fxopt/internal/model"
)

type Params struct {
	BaseMid  float64
	TickSize float64

	AnnualVol      float64 // average; actual volatility wanders around it
	VolLogSD       float64
	VolHalfLifeSec float64

	BaseSpread        float64
	SpreadLogSD       float64
	SpreadHalfLifeSec float64
	SpreadVolBeta     float64 // how hard spread reacts to volatility
	MinSpreadTicks    float64

	BaseLiquidity  float64 // base currency per second
	LiqVolBeta     float64 // how hard volume reacts to volatility; negative
	LiqLogSD       float64
	LiqHalfLifeSec float64

	JumpsPerSec float64
	JumpSizeSD  float64

	QuoteRefreshSec float64
}

func EURUSD() Params {
	return Params{
		BaseMid:  1.0850,
		TickSize: 0.00001,

		AnnualVol:      0.07,
		VolLogSD:       0.40,
		VolHalfLifeSec: 60,

		BaseSpread:        0.00010,
		SpreadLogSD:       0.30,
		SpreadHalfLifeSec: 20,
		SpreadVolBeta:     0.70,
		MinSpreadTicks:    1,

		BaseLiquidity:  6.0e6,
		LiqVolBeta:     -0.50,
		LiqLogSD:       0.25,
		LiqHalfLifeSec: 45,

		JumpsPerSec: 1.0e-4,
		JumpSizeSD:  0.0005,

		QuoteRefreshSec: 0.05,
	}
}

// GenerateMarket builds a series of market snapshots covering the horizon.
// Volatility is per year and liquidity per second, so asking for more slices
// samples the same market more finely, not a different one.
func GenerateMarket(p Params, slices int, horizon time.Duration, rng *rand.Rand) []model.MarketSlice {
	if slices <= 0 {
		return nil
	}

	dt := horizon.Seconds() / float64(slices)
	dtYears := dt / model.SecondsPerYear

	volState := newOU(p.VolLogSD, p.VolHalfLifeSec, dt, rng)
	spreadState := newOU(p.SpreadLogSD, p.SpreadHalfLifeSec, dt, rng)
	liqState := newOU(p.LiqLogSD, p.LiqHalfLifeSec, dt, rng)

	refreshEvery := 1
	if p.QuoteRefreshSec > dt {
		refreshEvery = int(math.Round(p.QuoteRefreshSec / dt))
	}

	minSpread := p.MinSpreadTicks * p.TickSize
	jumpProb := p.JumpsPerSec * dt

	market := make([]model.MarketSlice, slices)
	mid := p.BaseMid
	var spread, liquidity float64

	for i := range slices {
		volX := volState.next(rng)
		spreadX := spreadState.next(rng)
		liqX := liqState.next(rng)

		vol := p.AnnualVol * logNormalFactor(volX, p.VolLogSD)
		volRatio := vol / p.AnnualVol

		// The -sigma^2/2 stops the price creeping upward over time.
		sigma := vol * math.Sqrt(dtYears)
		logReturn := -0.5*sigma*sigma + sigma*rng.NormFloat64()
		if jumpProb > 0 && rng.Float64() < jumpProb {
			logReturn += p.JumpSizeSD * rng.NormFloat64()
		}
		mid = roundToTick(mid*math.Exp(logReturn), p.TickSize)

		// Quotes are stale between refreshes; the state processes still advance.
		if i%refreshEvery == 0 {
			spread = p.BaseSpread *
				logNormalFactor(spreadX, p.SpreadLogSD) *
				math.Pow(volRatio, p.SpreadVolBeta)
			spread = math.Max(roundToTick(spread, p.TickSize), minSpread)

			liquidity = p.BaseLiquidity *
				logNormalFactor(liqX, p.LiqLogSD) *
				math.Pow(volRatio, p.LiqVolBeta)
		}

		market[i] = model.MarketSlice{
			MidPrice:   mid,
			Spread:     spread,
			Liquidity:  liquidity,
			Volatility: vol,
		}
	}

	return market
}

// ou is a value that drifts at random but is always pulled back toward zero.
// Each step is worked out exactly, so its range doesn't depend on step size.
type ou struct {
	decay   float64
	shockSD float64
	x       float64
}

func newOU(statSD, halfLifeSec, dt float64, rng *rand.Rand) *ou {
	if statSD <= 0 {
		return &ou{}
	}

	o := &ou{shockSD: statSD}
	if halfLifeSec > 0 {
		o.decay = math.Exp(-math.Ln2 / halfLifeSec * dt)
		o.shockSD = statSD * math.Sqrt(1-o.decay*o.decay)
	}
	o.x = statSD * rng.NormFloat64() // start somewhere in its normal range
	return o
}

func (o *ou) next(rng *rand.Rand) float64 {
	if o.shockSD == 0 {
		return 0
	}
	o.x = o.decay*o.x + o.shockSD*rng.NormFloat64()
	return o.x
}

// logNormalFactor turns a log-space wobble into a multiplier averaging 1.
func logNormalFactor(x, sd float64) float64 {
	return math.Exp(x - sd*sd/2)
}

func roundToTick(v, tick float64) float64 {
	if tick <= 0 {
		return v
	}
	return math.Round(v/tick) * tick
}

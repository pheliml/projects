package sim

import (
	"math"
	"math/rand"

	"fxopt/internal/model"
)

func GenerateMarket(slices int, rng *rand.Rand) []model.MarketSlice {
	market := make([]model.MarketSlice, slices)

	baseMid := 1.0850 //EUR/USD
	baseSpread := 0.0001
	baseLiquidity := 2e6
	volatility := 0.0002

	mid := baseMid

	for i := 0; i < slices; i++ {
		shock := rng.NormFloat64() * volatility
		mid = mid * math.Exp(shock)

		// Add some noise
		spread := baseSpread * (0.8 + 0.4*rng.Float64())
		liquidity := baseLiquidity * (0.7 + 0.6*rng.Float64())

		market[i] = model.MarketSlice{
			MidPrice:   mid,
			Spread:     spread,
			Liquidity:  liquidity,
			Volatility: volatility,
		}
	}

	return market
}

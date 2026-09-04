package model

const SecondsPerYear = 252 * 24 * 3600

type MarketSlice struct {
	MidPrice   float64
	Spread     float64
	Liquidity  float64 // base currency per second
	Volatility float64 // annualised
}

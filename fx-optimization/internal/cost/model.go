package cost

import "fxopt/internal/model"

// impactCoeff scales the market impact contribution relative to volatility and participation.
const impactCoeff = 0.1

// ExecutionCost computes total execution cost in quote currency (e.g. USD for EUR/USD).
//
// Two components per slice:
//   - Spread cost: crossing half the bid-ask spread on every unit traded.
//   - Market impact: price slippage proportional to participation rate (qty / liquidity),
//     scaled by volatility and mid price — a simplified linear-in-participation model.
func ExecutionCost(order model.Order, schedule []float64, market []model.MarketSlice) float64 {
	totalCost := 0.0
	for i, qty := range schedule {
		ms := market[i]

		spreadCost := (ms.Spread / 2.0) * qty

		participation := qty / ms.Liquidity
		impactCost := impactCoeff * ms.Volatility * ms.MidPrice * qty * participation

		totalCost += spreadCost + impactCost
	}
	return totalCost
}

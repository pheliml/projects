package cost

import (
	"errors"
	"fmt"
	"math"

	"fxopt/internal/model"
)

type Params struct {
	PermanentCoeff float64 // Y in dP/P = Y * sigma_T * sqrt(X / V_T)
	TemporaryCoeff float64 // K in dP/P = K * sigma_T * participation
}

func EURUSDParams() Params {
	return Params{PermanentCoeff: 0.75, TemporaryCoeff: 1.0}
}

type Breakdown struct {
	Spread    float64
	Temporary float64
	Permanent float64
	Timing    float64 // drift of the unaffected mid from arrival; zero mean
	Total     float64

	Arrival  float64
	AvgPrice float64
	Quantity float64
}

func (b Breakdown) Bps() float64 {
	notional := b.Arrival * b.Quantity
	if notional == 0 {
		return 0
	}
	return b.Total / notional * 10_000
}

func Shortfall(p Params, order model.Order, schedule []float64, market []model.MarketSlice) (Breakdown, error) {
	if len(schedule) == 0 {
		return Breakdown{}, errors.New("cost: empty schedule")
	}
	if len(market) < len(schedule) {
		return Breakdown{}, fmt.Errorf("cost: market has %d slices, schedule needs %d", len(market), len(schedule))
	}

	dir := 1.0
	if order.Side == model.Sell {
		dir = -1.0
	}

	dt := order.Horizon().Seconds() / float64(len(schedule))
	arrival := market[0].MidPrice

	// Impact is normalised over the whole horizon, not per slice, so it depends
	// on the order's footprint
	quantity, sumVol, sumLiq := 0.0, 0.0, 0.0
	for i, qty := range schedule {
		quantity += qty
		sumVol += market[i].Volatility
		sumLiq += market[i].Liquidity
	}
	n := float64(len(schedule))
	horizonSec := dt * n
	sigmaT := sumVol / n * math.Sqrt(horizonSec/model.SecondsPerYear)
	volumeT := sumLiq / n * horizonSec

	fullImpact := 0.0
	if volumeT > 0 && quantity > 0 {
		fullImpact = arrival * p.PermanentCoeff * sigmaT * math.Sqrt(quantity/volumeT)
	}

	b := Breakdown{Arrival: arrival, Quantity: quantity}

	executed, execValue := 0.0, 0.0
	for i, qty := range schedule {
		ms := market[i]

		var permanent float64
		if quantity > 0 {
			permanent = fullImpact * math.Sqrt((executed+qty/2)/quantity)
		}

		var temporary float64
		if ms.Liquidity > 0 && dt > 0 {
			participation := qty / (ms.Liquidity * dt)
			temporary = arrival * p.TemporaryCoeff * sigmaT * participation
		}

		halfSpread := ms.Spread / 2
		execPrice := ms.MidPrice + dir*(permanent+halfSpread+temporary)

		b.Spread += halfSpread * qty
		b.Temporary += temporary * qty
		b.Permanent += permanent * qty
		b.Timing += dir * (ms.MidPrice - arrival) * qty

		executed += qty
		execValue += execPrice * qty
	}

	b.Total = b.Spread + b.Temporary + b.Permanent + b.Timing
	if quantity > 0 {
		b.AvgPrice = execValue / quantity
	}

	return b, nil
}

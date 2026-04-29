# FX Optimization

A Go program for simulating and evaluating FX execution strategies. Given an order (pair, notional, time horizon, and number of slices), it generates a synthetic market, runs an execution strategy, and calculates the total cost of execution.

## Project Structure

```
cmd/main.go                   - Entry point
internal/
  model/
    order.go                  - Order and Side types
    market.go                 - MarketSlice type (mid price, spread, liquidity, volatility)
  sim/
    marketgen.go              - Synthetic market generator
  strategy/
    twap.go                   - TWAP execution strategy
  cost/
    model.go                  - Execution cost calculation
```

## How It Works

### 1. Market Simulation

`sim.GenerateMarket` produces a slice of synthetic market snapshots using a log-normal mid price walk with randomised spread and liquidity noise. Each `MarketSlice` contains:

| Field        | Description                              |
|--------------|------------------------------------------|
| `MidPrice`   | Mid-market rate (e.g. 1.0850 for EUR/USD)|
| `Spread`     | Bid-ask spread in price terms            |
| `Liquidity`  | Available liquidity in base currency     |
| `Volatility` | Instantaneous volatility                 |

### 2. TWAP Strategy

`strategy.TWAP` implements a **Time-Weighted Average Price** schedule. It divides the total notional evenly across all time slices, producing a flat execution profile:

```
slice quantity = notional / slices
```

For a 10,000,000 EUR/USD order with 10 slices, each slice executes 1,000,000 units.

### 3. Execution Cost

`cost.ExecutionCost` computes the total cost in quote currency (e.g. USD) by summing two components across every slice:

**Spread cost** — the cost of crossing the bid-ask spread:
```
spread_cost = (spread / 2) × qty
```

**Market impact** — price slippage from trading against available liquidity, using a linear-in-participation model:
```
impact_cost = η × volatility × mid_price × qty × (qty / liquidity)
```

where `η = 0.1` is the impact coefficient. Larger slices relative to available liquidity produce proportionally higher slippage.

Total cost per slice:
```
cost = spread_cost + impact_cost
```

## Running

```bash
go run ./cmd/main.go
```

Example output:

```
------------------------
Pair:     EUR/USD
Side:     BUY
Notional: 10000000
Slices:   10

Execution schedule (TWAP):
  Slice  1: 1000000
  Slice  2: 1000000
  ...
  Slice 10: 1000000

Total execution cost: 620.45 USD
```

## Configuration

Order parameters are set directly in `cmd/main.go`:

| Field         | Description                        | Example      |
|---------------|------------------------------------|--------------|
| `Pair`        | Currency pair                      | `"EUR/USD"`  |
| `Side`        | Buy or Sell                        | `model.Buy`  |
| `Notional`    | Total order size in base currency  | `10_000_000` |
| `HorizonSec`  | Execution horizon in seconds       | `300`        |
| `Slices`      | Number of time slices              | `10`         |

// Package stats aggregates the results of repeated simulation runs.
package stats

import (
	"log/slog"
	"math"
)

const logPrecision = 4

type Summary struct {
	Count  int
	Mean   float64
	Min    float64
	Max    float64
	StdDev float64
}

// Summarise computes descriptive statistics for values.
// An empty input returns the zero Summary.
func Summarise(values []float64) Summary {
	if len(values) == 0 {
		return Summary{}
	}

	s := Summary{
		Count: len(values),
		Min:   values[0],
		Max:   values[0],
	}

	sum := 0.0
	for _, v := range values {
		sum += v
		s.Min = math.Min(s.Min, v)
		s.Max = math.Max(s.Max, v)
	}
	s.Mean = sum / float64(s.Count)

	// A single observation has no spread.
	if s.Count < 2 {
		return s
	}

	// Work out the mean first, then measure how far each value sits from it.
	sumSqDev := 0.0
	for _, v := range values {
		d := v - s.Mean
		sumSqDev += d * d
	}
	s.StdDev = math.Sqrt(sumSqDev / float64(s.Count-1))

	return s
}

// LogValue renders the Summary as a single slog attribute.
func (s Summary) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Int("count", s.Count),
		slog.Float64("mean", round(s.Mean)),
		slog.Float64("min", round(s.Min)),
		slog.Float64("max", round(s.Max)),
		slog.Float64("stddev", round(s.StdDev)),
	)
}

func round(v float64) float64 {
	scale := math.Pow(10, logPrecision)
	return math.Round(v*scale) / scale
}

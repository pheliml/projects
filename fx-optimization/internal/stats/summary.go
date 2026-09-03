// Package stats aggregates the results of repeated simulation runs.
package stats

import (
	"log/slog"
	"math"
)

// logPrecision is the number of decimal places retained when a Summary is logged.
// The Summary fields themselves are always exact.
const logPrecision = 4

// Summary describes the distribution of a set of observations.
type Summary struct {
	Count  int
	Mean   float64
	Min    float64
	Max    float64
	StdDev float64 // Sample standard deviation (Bessel-corrected); zero when Count < 2.
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

	// A single observation has no spread, and the n-1 denominator is undefined.
	if s.Count < 2 {
		return s
	}

	// Two-pass variance: subtracting the known mean first avoids the catastrophic
	// cancellation that a sum-of-squares accumulator suffers when the mean is
	// large relative to the spread — exactly the case here, where costs cluster
	// tightly around a few hundred USD.
	sumSqDev := 0.0
	for _, v := range values {
		d := v - s.Mean
		sumSqDev += d * d
	}
	s.StdDev = math.Sqrt(sumSqDev / float64(s.Count-1))

	return s
}

// LogValue renders the Summary as a slog group, so a call site can pass it as a
// single attribute value.
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

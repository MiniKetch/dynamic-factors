// df/rolling_stats.hpp — naive rolling mean/variance and z-score
// over a fixed-size window.
//
// Naming honesty: this is NOT a Welford-style streaming estimator. It
// uses running Σx and Σx² with the E[X²] − E[X]² identity, which is
// fine for our use case (log-return-scale numbers near zero) but would
// catastrophically cancel at price-scale magnitudes (think $50,000
// equity values squared). The negative-variance clamp at the bottom of
// the variance computation is the fingerprint of this method — it
// covers up the floating-point rounding that pushes the difference
// slightly below zero when all samples are nearly equal.
//
// We use these in two places:
//   1. Inside the signal generator, where we compute a rolling z-score
//      of each stock's residual against its own recent history.
//   2. Inside Kalman tuning, where we compute log-likelihood-equivalent
//      variances of innovations.
//
// All header-only because the templates are short and inlining matters.

#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>

namespace df {

/// Streaming mean / variance over a fixed-size rolling window.
///
/// Adds and drops samples in O(1) amortized — internally it uses a
/// deque to remember which sample to subtract when the window slides.
class RollingStats {
public:
    explicit RollingStats(std::size_t window) : window_(window) {
        if (window < 2) {
            throw std::invalid_argument(
                "RollingStats: window must be ≥ 2");
        }
    }

    /// Add a new sample. If the window is full, the oldest sample is
    /// removed first. Returns the current rolling z-score:
    ///     z = (sample - rolling_mean) / rolling_stdev
    /// or 0 if the window isn't yet full enough to estimate stdev.
    double push(double x) {
        // Drop oldest if the window is full.
        if (samples_.size() >= window_) {
            const double old = samples_.front();
            samples_.pop_front();
            sum_ -= old;
            sum_sq_ -= old * old;
        }
        samples_.push_back(x);
        sum_ += x;
        sum_sq_ += x * x;

        if (samples_.size() < 2) return 0.0;
        const double n = static_cast<double>(samples_.size());
        const double mean = sum_ / n;
        // Variance via E[X²] − E[X]² from running sums. This is the
        // numerically-naive form: when all samples are close together,
        // the two terms can be of similar magnitude and the difference
        // rounds slightly negative. Clamp to zero to absorb that.
        double var = (sum_sq_ - n * mean * mean) / (n - 1.0);
        if (var < 0.0) var = 0.0;
        const double sd = std::sqrt(var);
        if (!(sd > 0.0)) return 0.0;
        return (x - mean) / sd;
    }

    [[nodiscard]] std::size_t size()   const noexcept { return samples_.size(); }
    [[nodiscard]] std::size_t window() const noexcept { return window_; }

    [[nodiscard]] double mean() const noexcept {
        if (samples_.empty()) return 0.0;
        return sum_ / static_cast<double>(samples_.size());
    }

    [[nodiscard]] double variance() const noexcept {
        const auto n = samples_.size();
        if (n < 2) return 0.0;
        const double m = sum_ / static_cast<double>(n);
        double var = (sum_sq_ - static_cast<double>(n) * m * m)
                     / static_cast<double>(n - 1);
        return (var < 0.0) ? 0.0 : var;
    }

    void reset() noexcept {
        samples_.clear();
        sum_ = sum_sq_ = 0.0;
    }

private:
    std::size_t window_;
    std::deque<double> samples_;
    double sum_     = 0.0;
    double sum_sq_  = 0.0;
};

}  // namespace df

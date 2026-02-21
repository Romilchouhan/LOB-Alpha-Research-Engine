#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// RMSE & basic statistics (header-only, no Boost required)
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace stats {

/// Root Mean Squared Error between two equal-length series.
[[nodiscard]]
inline double rmse(const std::vector<double>& predicted,
                   const std::vector<double>& actual) {
    if (predicted.size() != actual.size() || predicted.empty()) return 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const double d = predicted[i] - actual[i];
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / static_cast<double>(predicted.size()));
}

/// Mean of a vector.
[[nodiscard]]
inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/// Standard deviation (population).
[[nodiscard]]
inline double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double sq = 0.0;
    for (auto x : v) sq += (x - m) * (x - m);
    return std::sqrt(sq / static_cast<double>(v.size()));
}

/// p-th percentile (0–100) of sorted data.
[[nodiscard]]
inline double percentile(std::vector<double> data, double p) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    const double idx = (p / 100.0) * static_cast<double>(data.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, data.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return data[lo] * (1.0 - frac) + data[hi] * frac;
}

} // namespace stats

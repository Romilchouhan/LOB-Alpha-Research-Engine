#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// ParquetWriter — feature-matrix export via Apache Arrow / Parquet
// ─────────────────────────────────────────────────────────────────────────────
//
// Column-buffered writer: append one FeatureRow per tick, then finalize()
// builds an Arrow table and writes a single Parquet file. Only compiled when
// LOB_ENABLE_PARQUET is ON (CMake defines LOB_HAS_PARQUET); this header keeps
// Arrow out of the include graph so lob_core headers stay light.
//
// Schema (23 columns):
//   tick (int64), mid, spread, micro_price, obi, ofi, ofi_rolling,
//   bid_slope, ask_slope, qimb_1..qimb_10, rv_50, rv_200, accel, vpin
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/price_level.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace io {

/// One row of the feature matrix (one LOB tick).
struct FeatureRow {
    std::int64_t tick        = 0;
    double       mid         = 0.0;
    double       spread      = 0.0;
    double       micro_price = 0.0;
    double       obi         = 0.0;
    double       ofi         = 0.0;   // per-event OFI
    double       ofi_rolling = 0.0;   // rolling backward-window sum
    double       bid_slope   = 0.0;
    double       ask_slope   = 0.0;
    std::array<double, lob::kMaxDepth> qimb{};
    double       rv_50       = 0.0;
    double       rv_200      = 0.0;
    double       accel       = 0.0;
    double       vpin        = 0.0;
};

class ParquetWriter {
public:
    explicit ParquetWriter(std::filesystem::path path);

    /// Buffer one row.
    void append(const FeatureRow& row);

    /// Build the Arrow table and write the Parquet file.
    /// Throws std::runtime_error on I/O or Arrow failure. Idempotent no-op
    /// after the first successful call.
    void finalize();

    /// Number of rows buffered so far.
    [[nodiscard]] std::size_t rows() const noexcept { return tick_.size(); }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    bool                  finalized_ = false;

    std::vector<std::int64_t> tick_;
    std::vector<double> mid_, spread_, micro_price_, obi_, ofi_, ofi_rolling_;
    std::vector<double> bid_slope_, ask_slope_;
    std::array<std::vector<double>, lob::kMaxDepth> qimb_;
    std::vector<double> rv_50_, rv_200_, accel_, vpin_;
};

} // namespace io

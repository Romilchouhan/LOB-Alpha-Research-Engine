#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// FI-2010 CSV & Binary Parser
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/price_level.hpp"
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace lob {

/// Parse a FI-2010–format CSV file into a vector of LOBSnapshots.
/// Expected column layout (per row, 40 values):
///   ask_price_1 … ask_price_10, ask_vol_1 … ask_vol_10,
///   bid_price_1 … bid_price_10, bid_vol_1 … bid_vol_10
///
/// Lines starting with '#' or alphabetic chars are treated as headers / comments.
[[nodiscard]]
std::vector<LOBSnapshot> parse_fi2010_csv(const std::filesystem::path& file);

/// Read a pre-processed binary file (written by preprocess.py).
/// Layout: N × sizeof(LOBSnapshot) contiguous bytes, little-endian doubles.
[[nodiscard]]
std::vector<LOBSnapshot> read_binary(const std::filesystem::path& file);

/// Generate a small synthetic dataset for unit / smoke testing.
[[nodiscard]]
std::vector<LOBSnapshot> generate_synthetic(std::size_t n = 500);

} // namespace lob

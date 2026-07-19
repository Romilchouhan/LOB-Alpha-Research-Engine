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
/// Expected column layout (per row, first 40 values) is PER-LEVEL INTERLEAVED,
/// matching the canonical FI-2010 release. For level L (0-based, levels 1..10):
///   vals[4L+0] = ask_price_(L+1)   vals[4L+1] = ask_vol_(L+1)
///   vals[4L+2] = bid_price_(L+1)   vals[4L+3] = bid_vol_(L+1)
/// i.e. [Pa1,Va1,Pb1,Vb1, Pa2,Va2,Pb2,Vb2, …, Pa10,Va10,Pb10,Vb10].
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

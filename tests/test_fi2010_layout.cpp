// ─────────────────────────────────────────────────────────────────────────────
// Regression tests — FI-2010 PER-LEVEL INTERLEAVED column layout.
//
// The raw FI-2010 file lays out the first 40 LOB columns per-level interleaved:
//   for level L (0-based, levels 1..10):
//     vals[4L+0] = ask_price   vals[4L+1] = ask_volume
//     vals[4L+2] = bid_price   vals[4L+3] = bid_volume
//
// A prior bug assumed a BLOCK layout (ask_px[0:10], ask_vol[10:20], …) which
// scrambled every derived feature. These tests pin the corrected mapping.
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/fi2010_parser.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace {

/// Serialize 40 doubles as a single CSV row (no trailing comma), at full
/// round-trip precision so parsed values equal the source literals exactly.
std::string csv_row(const double (&v)[lob::kFI2010Columns]) {
    std::ostringstream os;
    os.precision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < lob::kFI2010Columns; ++i) {
        if (i) os << ',';
        os << v[i];
    }
    return os.str();
}

std::vector<lob::LOBSnapshot> parse_single_row(
        const std::string& row, const char* name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    {
        std::ofstream out(path);
        out << row << "\n";
    }
    auto snaps = lob::parse_fi2010_csv(path);
    std::filesystem::remove(path);
    return snaps;
}

} // namespace

// ── Hand-computed numeric case ────────────────────────────────────────────────
// Build a physically valid book directly in the interleaved layout with unique
// values per slot, so an exact decode is the only way every assertion passes.
TEST(FI2010Layout, InterleavedDecodeHandComputed) {
    double v[lob::kFI2010Columns]{};
    for (int L = 0; L < lob::kMaxDepth; ++L) {
        v[4 * L + 0] = 100.0 + 0.10 * L;   // ask price  — strictly increasing
        v[4 * L + 1] = 10.0 + L;           // ask volume
        v[4 * L + 2] = 99.0 - 0.10 * L;    // bid price  — strictly decreasing
        v[4 * L + 3] = 20.0 + L;           // bid volume
    }

    const auto snaps = parse_single_row(csv_row(v), "lob_interleaved_hand.csv");
    ASSERT_EQ(snaps.size(), 1u);
    const auto& s = snaps[0];

    // Exact mapping for level 1 (hand-computed from the raw slots above).
    EXPECT_DOUBLE_EQ(s.asks[0].price, 100.0);   // v[0]
    EXPECT_DOUBLE_EQ(s.asks[0].volume, 10.0);   // v[1]
    EXPECT_DOUBLE_EQ(s.bids[0].price, 99.0);    // v[2]
    EXPECT_DOUBLE_EQ(s.bids[0].volume, 20.0);   // v[3]

    // Level 3 (raw slots 8..11) — guards against off-by-one in the stride.
    EXPECT_DOUBLE_EQ(s.asks[2].price, 100.2);   // v[8]
    EXPECT_DOUBLE_EQ(s.asks[2].volume, 12.0);   // v[9]
    EXPECT_DOUBLE_EQ(s.bids[2].price, 98.8);    // v[10]
    EXPECT_DOUBLE_EQ(s.bids[2].volume, 22.0);   // v[11]

    // Deepest level 10 (raw slots 36..39).
    EXPECT_DOUBLE_EQ(s.asks[9].price, 100.9);   // v[36]
    EXPECT_DOUBLE_EQ(s.asks[9].volume, 19.0);   // v[37]
    EXPECT_DOUBLE_EQ(s.bids[9].price, 98.1);    // v[38]
    EXPECT_DOUBLE_EQ(s.bids[9].volume, 29.0);   // v[39]

    // Sign correctness: positive spread and monotone book on a valid input.
    EXPECT_GT(s.asks[0].price, s.bids[0].price);
    for (int i = 1; i < lob::kMaxDepth; ++i) {
        EXPECT_GT(s.asks[i].price, s.asks[i - 1].price);
        EXPECT_LT(s.bids[i].price, s.bids[i - 1].price);
    }
}

// ── Known real raw row (row 0 of data/FI2010_train.csv, Zscore variant) ───────
// Structural checks that hold for this specific normalized row: even raw slots
// are prices (>0 here), odd raw slots are volumes (<0 under Zscore), and the
// best-level spread is positive. NOTE: on Zscore data these do NOT hold for
// every row — each column is standardized independently, so global monotonicity
// / spread sign is not guaranteed (that needs the un-normalized DecPre variant).
TEST(FI2010Layout, RealZscoreRowStructure) {
    const double v[lob::kFI2010Columns] = {
        0.318116, -0.56461858, 0.31353946, -0.551889,
        0.31972639, -0.73122754, 0.31289065, -0.42544781,
        0.31940378, -0.8441572, 0.31225423, -0.57176602,
        0.31913811, -0.76965943, 0.31154589, -0.61917365,
        0.31888116, -0.95674446, 0.31181907, -0.88562676,
        0.3194785, -0.49287635, 0.30925662, -0.80712815,
        0.31991981, -0.70386572, 0.30881154, -0.7419996,
        0.31920178, -0.35745113, 0.30856466, -0.62497773,
        0.32024158, -0.48331295, 0.30752602, -0.53321812,
        0.32206076, -0.42776259, 0.300703, -0.48082799,
    };

    const auto snaps = parse_single_row(csv_row(v), "lob_interleaved_real.csv");
    ASSERT_EQ(snaps.size(), 1u);
    const auto& s = snaps[0];

    // Even raw slots decode to prices; every price on this row is positive.
    for (int i = 0; i < lob::kMaxDepth; ++i) {
        EXPECT_GT(s.asks[i].price, 0.0);
        EXPECT_GT(s.bids[i].price, 0.0);
    }
    // Odd raw slots decode to volumes; all negative on this Zscore row.
    for (int i = 0; i < lob::kMaxDepth; ++i) {
        EXPECT_LT(s.asks[i].volume, 0.0);
        EXPECT_LT(s.bids[i].volume, 0.0);
    }
    // Positive best-level spread for this known row.
    EXPECT_GT(s.asks[0].price, s.bids[0].price);
    EXPECT_DOUBLE_EQ(s.asks[0].price, 0.318116);
    EXPECT_DOUBLE_EQ(s.bids[0].price, 0.31353946);
}

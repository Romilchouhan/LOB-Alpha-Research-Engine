// ─────────────────────────────────────────────────────────────────────────────
// Tests — LimitOrderBook invariants & FI-2010 parser
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/fi2010_parser.hpp"
#include "lob/order_book.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

// ── LOB invariants over synthetic data ──────────────────────────────────────

TEST(LimitOrderBook, SyntheticDataInvariants) {
    const auto snaps = lob::generate_synthetic(500);
    ASSERT_EQ(snaps.size(), 500u);

    lob::LimitOrderBook book;
    for (const auto& snap : snaps) {
        book.update(snap);

        // Spread ≥ 0 and mid strictly inside [best_bid, best_ask].
        ASSERT_GE(book.spread(), 0.0);
        ASSERT_GE(book.mid_price(), book.best_bid().price);
        ASSERT_LE(book.mid_price(), book.best_ask().price);

        // Bids strictly decreasing, asks strictly increasing with depth.
        for (int i = 1; i < lob::kMaxDepth; ++i) {
            ASSERT_LT(snap.bids[i].price, snap.bids[i - 1].price);
            ASSERT_GT(snap.asks[i].price, snap.asks[i - 1].price);
        }
    }
    EXPECT_EQ(book.tick_count(), 500u);
}

TEST(LimitOrderBook, DerivedQuantities) {
    lob::LOBSnapshot snap{};
    snap.bids[0] = {100.0, 30.0};
    snap.asks[0] = {101.0, 10.0};
    snap.bids[1] = {99.0, 5.0};
    snap.asks[1] = {102.0, 5.0};

    lob::LimitOrderBook book;
    book.update(snap);

    EXPECT_DOUBLE_EQ(book.mid_price(), 100.5);
    EXPECT_DOUBLE_EQ(book.spread(), 1.0);
    EXPECT_DOUBLE_EQ(book.total_volume_bid(), 35.0);
    EXPECT_DOUBLE_EQ(book.total_volume_ask(), 15.0);
    EXPECT_DOUBLE_EQ(book.total_volume_bid(1), 30.0);
    EXPECT_DOUBLE_EQ(book.total_volume_ask(1), 10.0);
}

// ── FI-2010 CSV parser ──────────────────────────────────────────────────────

namespace {

/// Build one 40-column FI-2010 row in the canonical PER-LEVEL INTERLEAVED
/// layout: [P_ask, V_ask, P_bid, V_bid] repeated for levels 1..10.
std::string make_row(double base) {
    std::string row;
    for (int i = 0; i < 10; ++i) {                     // level i (0-based)
        row += std::to_string(base + 0.5 + 0.1 * i) + ",";  // ask price
        row += std::to_string(100.0 + i) + ",";            // ask volume
        row += std::to_string(base - 0.5 - 0.1 * i) + ",";  // bid price
        row += std::to_string(200.0 + i);                  // bid volume
        if (i < 9) row += ",";
    }
    return row;
}

} // namespace

TEST(FI2010Parser, ParsesTinyCsvFixture) {
    const auto path = std::filesystem::temp_directory_path()
                      / "lob_test_fixture.csv";
    {
        std::ofstream out(path);
        out << "# header comment line\n";
        out << make_row(100.0) << "\n";
        out << make_row(200.0) << "\n";
    }

    const auto snaps = lob::parse_fi2010_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(snaps.size(), 2u);

    // Row 1: base 100
    EXPECT_DOUBLE_EQ(snaps[0].asks[0].price, 100.5);
    EXPECT_DOUBLE_EQ(snaps[0].asks[0].volume, 100.0);
    EXPECT_DOUBLE_EQ(snaps[0].bids[0].price, 99.5);
    EXPECT_DOUBLE_EQ(snaps[0].bids[0].volume, 200.0);

    // Deep levels of row 1
    EXPECT_DOUBLE_EQ(snaps[0].asks[9].price, 100.5 + 0.9);
    EXPECT_DOUBLE_EQ(snaps[0].asks[9].volume, 109.0);
    EXPECT_DOUBLE_EQ(snaps[0].bids[9].price, 99.5 - 0.9);
    EXPECT_DOUBLE_EQ(snaps[0].bids[9].volume, 209.0);

    // Row 2: base 200
    EXPECT_DOUBLE_EQ(snaps[1].asks[0].price, 200.5);
    EXPECT_DOUBLE_EQ(snaps[1].bids[0].price, 199.5);
}

TEST(FI2010Parser, SkipsMalformedRows) {
    const auto path = std::filesystem::temp_directory_path()
                      / "lob_test_malformed.csv";
    {
        std::ofstream out(path);
        out << "1.0,2.0,3.0\n";           // too few columns — skipped
        out << make_row(100.0) << "\n";
    }

    const auto snaps = lob::parse_fi2010_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(snaps.size(), 1u);
    EXPECT_DOUBLE_EQ(snaps[0].asks[0].price, 100.5);
}

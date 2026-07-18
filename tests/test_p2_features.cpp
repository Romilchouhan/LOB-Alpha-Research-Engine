// ─────────────────────────────────────────────────────────────────────────────
// Tests — P2 features: OFI, BookSlope, QueueImbalance, RealizedVol, Accel,
//          Parquet round-trip
// ─────────────────────────────────────────────────────────────────────────────

#include "features/accel.hpp"
#include "features/book_slope.hpp"
#include "features/ofi.hpp"
#include "features/queue_imbalance.hpp"
#include "features/realized_vol.hpp"
#include "lob/order_book.hpp"
#include "stats/rmse.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#ifdef LOB_HAS_PARQUET
#include "io/parquet_writer.hpp"
// Workaround for Apple SDKs that ship std::bit_width without defining
// __cpp_lib_bitops (see src/parquet_writer.cpp).
#include <bit>
#if defined(__apple_build_version__) && !defined(__cpp_lib_bitops)
#define __cpp_lib_bitops 201907L
#endif
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#endif

namespace {

/// Build a book from explicit per-level (price, volume) lists; unlisted
/// levels stay {0, 0} (unpopulated).
lob::LimitOrderBook make_deep_book(
    const std::vector<lob::PriceLevel>& bids,
    const std::vector<lob::PriceLevel>& asks) {
    lob::LOBSnapshot snap{};
    for (std::size_t i = 0; i < bids.size() && i < static_cast<std::size_t>(lob::kMaxDepth); ++i)
        snap.bids[i] = bids[i];
    for (std::size_t i = 0; i < asks.size() && i < static_cast<std::size_t>(lob::kMaxDepth); ++i)
        snap.asks[i] = asks[i];
    lob::LimitOrderBook book;
    book.update(snap);
    return book;
}

} // namespace

// ── Order Flow Imbalance ────────────────────────────────────────────────────

TEST(Ofi, FirstEventIsZero) {
    features::OrderFlowImbalance ofi(10);
    // No previous snapshot → e_1 = 0 by definition.
    EXPECT_DOUBLE_EQ(ofi.update(100.0, 10.0, 101.0, 20.0), 0.0);
    EXPECT_DOUBLE_EQ(ofi.rolling_sum(), 0.0);
}

TEST(Ofi, BidPriceUpIsPositive) {
    // prev: bid 100 @ 10, ask 101 @ 20 → next: bid 100.5 @ 8, ask 101 @ 20
    //   bid: 100.5 ≥ 100 → +q_b,n = +8 ; 100.5 ≤ 100 false → no −10
    //   ask: 101 ≤ 101 → −q_a,n = −20 ; 101 ≥ 101 → +q_a,n−1 = +20  (net 0)
    //   e = 8 − 20 + 20 = 8
    features::OrderFlowImbalance ofi(10);
    (void)ofi.update(100.0, 10.0, 101.0, 20.0);
    EXPECT_DOUBLE_EQ(ofi.update(100.5, 8.0, 101.0, 20.0), 8.0);
}

TEST(Ofi, AskSizeIncreaseAtSamePriceIsNegative) {
    // prev: bid 100 @ 10, ask 101 @ 20 → next: bid 100 @ 10, ask 101 @ 35
    //   bid: price flat → +10 − 10 = 0
    //   ask: price flat → −q_a,n + q_a,n−1 = −35 + 20 = −15  (sell pressure)
    //   e = −15
    features::OrderFlowImbalance ofi(10);
    (void)ofi.update(100.0, 10.0, 101.0, 20.0);
    EXPECT_DOUBLE_EQ(ofi.update(100.0, 10.0, 101.0, 35.0), -15.0);
}

TEST(Ofi, BidPriceDownIsNegative) {
    // prev: bid 100 @ 10, ask 101 @ 20 → next: bid 99.5 @ 12, ask 101 @ 20
    //   bid: 99.5 ≥ 100 false → no +12 ; 99.5 ≤ 100 → −q_b,n−1 = −10
    //   ask: flat → 0
    //   e = −10
    features::OrderFlowImbalance ofi(10);
    (void)ofi.update(100.0, 10.0, 101.0, 20.0);
    EXPECT_DOUBLE_EQ(ofi.update(99.5, 12.0, 101.0, 20.0), -10.0);
}

TEST(Ofi, BidSizeIncreaseAtSamePriceIsPositive) {
    // prev: bid 100 @ 10 → next: bid 100 @ 25 (ask unchanged)
    //   bid: flat → +25 − 10 = +15 ; ask flat → 0 ; e = +15
    features::OrderFlowImbalance ofi(10);
    (void)ofi.update(100.0, 10.0, 101.0, 20.0);
    EXPECT_DOUBLE_EQ(ofi.update(100.0, 25.0, 101.0, 20.0), 15.0);
}

TEST(Ofi, RollingSumIsBackwardWindow) {
    // window = 2 → rolling sum holds only the last two events.
    features::OrderFlowImbalance ofi(2);
    (void)ofi.update(100.0, 10.0, 101.0, 20.0);            // e1 = 0
    const double e2 = ofi.update(100.0, 25.0, 101.0, 20.0); // +15 (bid add)
    EXPECT_DOUBLE_EQ(ofi.rolling_sum(), 0.0 + e2);
    const double e3 = ofi.update(100.0, 25.0, 101.0, 35.0); // −15 (ask add)
    EXPECT_DOUBLE_EQ(ofi.rolling_sum(), e2 + e3);            // e1 evicted
    const double e4 = ofi.update(100.5, 8.0, 101.0, 35.0);   // +8 (bid up)
    EXPECT_DOUBLE_EQ(ofi.rolling_sum(), e3 + e4);            // e2 evicted
    EXPECT_DOUBLE_EQ(ofi.value(), e4);
}

// ── Book Slope ──────────────────────────────────────────────────────────────

TEST(BookSlope, HandComputedLinearBook) {
    // mid = (99 + 101)/2 = 100.
    // Bids: (99,10) (98,10) (97,10) → x = {1,2,3}, cum y = {10,20,30}
    //   x̄ = 2, ȳ = 20 ; Sxy = (−1)(−10)+(0)(0)+(1)(10) = 20 ; Sxx = 1+0+1 = 2
    //   bid_slope = 20/2 = 10
    // Asks: (101,5) (102,5) (103,5) → x = {1,2,3}, cum y = {5,10,15}
    //   Sxy = 10, Sxx = 2 → ask_slope = 5
    const auto book = make_deep_book(
        {{99.0, 10.0}, {98.0, 10.0}, {97.0, 10.0}},
        {{101.0, 5.0}, {102.0, 5.0}, {103.0, 5.0}});
    const auto s = features::BookSlope::compute(book);
    EXPECT_DOUBLE_EQ(s.bid_slope, 10.0);
    EXPECT_DOUBLE_EQ(s.ask_slope, 5.0);
}

TEST(BookSlope, DegenerateSingleLevelIsZero) {
    // Only one populated level per side → cannot regress → 0 by contract.
    const auto book = make_deep_book({{99.0, 10.0}}, {{101.0, 5.0}});
    const auto s = features::BookSlope::compute(book);
    EXPECT_DOUBLE_EQ(s.bid_slope, 0.0);
    EXPECT_DOUBLE_EQ(s.ask_slope, 0.0);
}

// ── Queue Imbalance ─────────────────────────────────────────────────────────

TEST(QueueImbalance, HandComputedAndZeroDenominator) {
    // Level 1: (30 − 10)/40 = 0.5
    // Level 2: (0 − 20)/20  = −1
    // Level 3+: both sides empty → 0/0 → defined as 0
    const auto book = make_deep_book(
        {{100.0, 30.0}, {99.0, 0.0}},
        {{101.0, 10.0}, {102.0, 20.0}});
    const auto qi = features::QueueImbalance::compute(book);
    EXPECT_DOUBLE_EQ(qi[0], 0.5);
    EXPECT_DOUBLE_EQ(qi[1], -1.0);
    for (std::size_t i = 2; i < static_cast<std::size_t>(lob::kMaxDepth); ++i)
        EXPECT_DOUBLE_EQ(qi[i], 0.0);
}

// ── Realized Volatility ─────────────────────────────────────────────────────

TEST(RealizedVol, HandComputedWindow2AndWarmupNaN) {
    // mids = {100, 102, 101, 103}, window W = 2
    //   r1 = ln(102/100), r2 = ln(101/102), r3 = ln(103/101)
    // tick 1: no return          → NaN (warm-up)
    // tick 2: 1 return  (< W)    → NaN (warm-up)
    // tick 3: {r1, r2} → population stdev = |r1 − r2| / 2
    //   = (0.0198026273… + 0.0098522964…)/2 = 0.0148274619…
    // tick 4: {r2, r3} → (r3 − r2)/2 = (0.0196084550… + 0.0098522964…)/2
    features::RealizedVolatility rv(2);
    EXPECT_TRUE(std::isnan(rv.update(100.0)));
    EXPECT_TRUE(std::isnan(rv.update(102.0)));

    const double r1 = std::log(102.0 / 100.0);
    const double r2 = std::log(101.0 / 102.0);
    const double r3 = std::log(103.0 / 101.0);

    const double v3 = rv.update(101.0);
    EXPECT_NEAR(v3, (r1 - r2) / 2.0, 1e-15);
    EXPECT_NEAR(v3, 0.014827461886, 1e-10);           // hand value
    EXPECT_NEAR(v3, stats::stddev({r1, r2}), 1e-15);  // reuse stats helper

    const double v4 = rv.update(103.0);
    EXPECT_NEAR(v4, (r3 - r2) / 2.0, 1e-15);
    EXPECT_NEAR(v4, stats::stddev({r2, r3}), 1e-15);
}

TEST(RealizedVol, StrictlyBackwardLooking) {
    // The value at tick t must depend only on mids[0..t]: feeding a huge
    // spike AFTER tick t must not change what was reported at tick t.
    const std::vector<double> mids{100, 101, 100.5, 101.5, 100.8, 101.2};

    features::RealizedVolatility full(3);
    std::vector<double> full_vals;
    full_vals.reserve(mids.size());
    for (double m : mids) full_vals.push_back(full.update(m));
    (void)full.update(500.0);   // future spike — irrelevant to recorded values

    for (std::size_t t = 0; t < mids.size(); ++t) {
        features::RealizedVolatility prefix(3);
        double v = 0.0;
        for (std::size_t k = 0; k <= t; ++k) v = prefix.update(mids[k]);
        if (std::isnan(full_vals[t]))
            EXPECT_TRUE(std::isnan(v)) << "tick " << t;
        else
            EXPECT_DOUBLE_EQ(full_vals[t], v) << "tick " << t;
    }
}

TEST(RealizedVol, ConstantMidGivesZeroAfterWarmup) {
    features::RealizedVolatility rv(3);
    (void)rv.update(100.0);
    (void)rv.update(100.0);
    (void)rv.update(100.0);
    EXPECT_DOUBLE_EQ(rv.update(100.0), 0.0);   // 3 zero-returns → σ = 0
}

// ── Mid-Price Acceleration ──────────────────────────────────────────────────

TEST(Accel, HandComputedSecondDifference) {
    // mids = {100, 101, 103, 102}
    //   t1, t2 → NaN (need three points)
    //   t3: 103 − 2·101 + 100 = 1
    //   t4: 102 − 2·103 + 101 = −3
    features::MidAcceleration accel;
    EXPECT_TRUE(std::isnan(accel.update(100.0)));
    EXPECT_TRUE(std::isnan(accel.update(101.0)));
    EXPECT_DOUBLE_EQ(accel.update(103.0), 1.0);
    EXPECT_DOUBLE_EQ(accel.update(102.0), -3.0);
    EXPECT_DOUBLE_EQ(accel.value(), -3.0);
}

// ── Parquet round-trip ──────────────────────────────────────────────────────

TEST(Parquet, RoundTrip) {
#ifndef LOB_HAS_PARQUET
    GTEST_SKIP() << "Built without Arrow/Parquet (LOB_ENABLE_PARQUET=OFF)";
#else
    const auto path = std::filesystem::temp_directory_path()
                    / "lob_p2_roundtrip.parquet";

    {
        io::ParquetWriter w(path);
        for (int t = 0; t < 3; ++t) {
            io::FeatureRow row;
            row.tick        = t;
            row.mid         = 100.0 + t;
            row.spread      = 0.5;
            row.micro_price = 100.25 + t;
            row.obi         = 0.1 * t;
            row.ofi         = 5.0 * t;
            row.ofi_rolling = 5.0 * t + 1.0;
            row.bid_slope   = 10.0;
            row.ask_slope   = 5.0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(lob::kMaxDepth); ++i)
                row.qimb[i] = 0.01 * static_cast<double>(i) + 0.1 * t;
            row.rv_50  = (t == 0) ? std::numeric_limits<double>::quiet_NaN()
                                  : 0.001 * t;
            row.rv_200 = std::numeric_limits<double>::quiet_NaN();
            row.accel  = -1.0 + t;
            row.dif   = 0.4;
            w.append(row);
        }
        ASSERT_EQ(w.rows(), 3u);
        w.finalize();
    }

    auto infile_res = arrow::io::ReadableFile::Open(path.string());
    ASSERT_TRUE(infile_res.ok()) << infile_res.status().ToString();

    auto reader_res = parquet::arrow::OpenFile(*infile_res,
                                               arrow::default_memory_pool());
    ASSERT_TRUE(reader_res.ok()) << reader_res.status().ToString();
    auto reader = std::move(reader_res).ValueOrDie();

    auto table_res = reader->ReadTable();
    ASSERT_TRUE(table_res.ok()) << table_res.status().ToString();
    const std::shared_ptr<arrow::Table> table = std::move(table_res).ValueOrDie();

    EXPECT_EQ(table->num_rows(), 3);
    EXPECT_EQ(table->num_columns(), 23);

    const auto col_double = [&](const std::string& name, int row) {
        const auto col = table->GetColumnByName(name);
        EXPECT_NE(col, nullptr) << "missing column " << name;
        const auto arr =
            std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
        return arr->Value(row);
    };

    // tick column (int64)
    const auto tick_arr = std::static_pointer_cast<arrow::Int64Array>(
        table->GetColumnByName("tick")->chunk(0));
    EXPECT_EQ(tick_arr->Value(0), 0);
    EXPECT_EQ(tick_arr->Value(2), 2);

    EXPECT_DOUBLE_EQ(col_double("mid", 1), 101.0);
    EXPECT_DOUBLE_EQ(col_double("micro_price", 2), 102.25);
    EXPECT_DOUBLE_EQ(col_double("ofi", 2), 10.0);
    EXPECT_DOUBLE_EQ(col_double("ofi_rolling", 1), 6.0);
    EXPECT_DOUBLE_EQ(col_double("bid_slope", 0), 10.0);
    EXPECT_DOUBLE_EQ(col_double("qimb_1", 1), 0.1);       // i=0, t=1
    EXPECT_DOUBLE_EQ(col_double("qimb_10", 0), 0.09);     // i=9, t=0
    EXPECT_TRUE(std::isnan(col_double("rv_50", 0)));      // warm-up NaN survives
    EXPECT_DOUBLE_EQ(col_double("rv_50", 2), 0.002);
    EXPECT_DOUBLE_EQ(col_double("accel", 0), -1.0);
    EXPECT_DOUBLE_EQ(col_double("depth_imb_flow", 2), 0.4);

    std::filesystem::remove(path);
#endif
}

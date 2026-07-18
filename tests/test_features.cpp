// ─────────────────────────────────────────────────────────────────────────────
// Tests — MicroPrice, OBI, DepthImbalanceFlow, stats helpers
// ─────────────────────────────────────────────────────────────────────────────

#include "features/micro_price.hpp"
#include "features/obi.hpp"
#include "features/depth_imbalance_flow.hpp"
#include "lob/fi2010_parser.hpp"
#include "lob/order_book.hpp"
#include "stats/rmse.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

/// Build a book with a single populated level on each side.
lob::LimitOrderBook make_book(double bid_px, double bid_vol,
                              double ask_px, double ask_vol) {
    lob::LOBSnapshot snap{};
    snap.bids[0] = {bid_px, bid_vol};
    snap.asks[0] = {ask_px, ask_vol};
    lob::LimitOrderBook book;
    book.update(snap);
    return book;
}

} // namespace

// ── MicroPrice ──────────────────────────────────────────────────────────────

TEST(MicroPrice, HandComputedOneLevelCase) {
    // Stoikov: μ = P_ask·(V_bid/(V_bid+V_ask)) + P_bid·(V_ask/(V_bid+V_ask))
    // bid = 100 @ 30, ask = 101 @ 10:
    //   μ = 101·(30/40) + 100·(10/40) = 75.75 + 25.0 = 100.75
    const auto book = make_book(100.0, 30.0, 101.0, 10.0);
    EXPECT_DOUBLE_EQ(features::MicroPrice::compute(book), 100.75);
}

TEST(MicroPrice, HeavierAskVolumePushesTowardBid) {
    // Ask side dominates → selling pressure → micro-price below mid.
    const auto book = make_book(100.0, 10.0, 101.0, 30.0);
    const double micro = features::MicroPrice::compute(book);
    EXPECT_LT(micro, book.mid_price());
    // μ = 101·(10/40) + 100·(30/40) = 25.25 + 75.0 = 100.25
    EXPECT_DOUBLE_EQ(micro, 100.25);
}

TEST(MicroPrice, ZeroVolumeFallsBackToMid) {
    // No BBO volume — must not divide by zero; fall back to mid.
    const auto book = make_book(100.0, 0.0, 101.0, 0.0);
    EXPECT_DOUBLE_EQ(features::MicroPrice::compute(book), 100.5);
}

// ── Order Book Imbalance ────────────────────────────────────────────────────

TEST(OrderBookImbalance, HandComputedCase) {
    // OBI = (V_bid − V_ask) / (V_bid + V_ask) = (30 − 10) / 40 = 0.5
    const auto book = make_book(100.0, 30.0, 101.0, 10.0);
    EXPECT_DOUBLE_EQ(features::OrderBookImbalance::compute(book), 0.5);
}

TEST(OrderBookImbalance, SignCorrectness) {
    // Bid-heavy → positive; ask-heavy → negative.
    EXPECT_GT(features::OrderBookImbalance::compute(
                  make_book(100.0, 50.0, 101.0, 10.0)), 0.0);
    EXPECT_LT(features::OrderBookImbalance::compute(
                  make_book(100.0, 10.0, 101.0, 50.0)), 0.0);
}

TEST(OrderBookImbalance, Bounds) {
    // All volume on one side → exactly ±1; always within [−1, 1].
    EXPECT_DOUBLE_EQ(features::OrderBookImbalance::compute(
                         make_book(100.0, 50.0, 101.0, 0.0)), 1.0);
    EXPECT_DOUBLE_EQ(features::OrderBookImbalance::compute(
                         make_book(100.0, 0.0, 101.0, 50.0)), -1.0);

    for (const auto& snap : lob::generate_synthetic(100)) {
        lob::LimitOrderBook book;
        book.update(snap);
        const double obi = features::OrderBookImbalance::compute(book);
        ASSERT_GE(obi, -1.0);
        ASSERT_LE(obi, 1.0);
    }
    // Zero volume everywhere → defined result of 0.
    EXPECT_DOUBLE_EQ(features::OrderBookImbalance::compute(
                         make_book(100.0, 0.0, 101.0, 0.0)), 0.0);
}

// ── DepthImbalanceFlow ────────────────────────────────────────────────────────────────────

TEST(Vpin, BucketAccumulationAndFlush) {
    // bucket_volume = 100, window = 2 buckets.
    features::DepthImbalanceFlow dif(100.0, 2);

    // Tick 1 (first tick → 50/50 split): fills bucket 1 = {50 buy, 50 sell}.
    // Only 1 bucket in window → DepthImbalanceFlow still NaN.
    EXPECT_TRUE(std::isnan(dif.update(100.0, 10.0)));

    // Tick 2 (mid up → all buy): bucket 2 = {100, 0}.
    // Window full: DepthImbalanceFlow = (|50−50| + |100−0|) / (2·100) = 0.5
    EXPECT_DOUBLE_EQ(dif.update(100.0, 11.0), 0.5);
    EXPECT_DOUBLE_EQ(dif.value(), 0.5);

    // Tick 3 (mid down → all sell): bucket 3 = {0, 100}, bucket 1 evicted.
    // DepthImbalanceFlow = (|100−0| + |0−100|) / 200 = 1.0
    EXPECT_DOUBLE_EQ(dif.update(100.0, 10.0), 1.0);

    EXPECT_EQ(dif.history().size(), 2u);
}

TEST(Vpin, PartialBucketWithOverflowRatio) {
    // bucket_volume = 100, window = 1 bucket.
    features::DepthImbalanceFlow dif(100.0, 1);

    // Tick 1: 60 vol, first tick → 30/30. Bucket not full → NaN.
    EXPECT_TRUE(std::isnan(dif.update(60.0, 10.0)));

    // Tick 2: 60 vol, mid up → all buy. Accumulated: buy 90, sell 30, tot 120.
    // Flush ratio = 100/120 → bucket = {75, 25}; DepthImbalanceFlow = |75−25|/100 = 0.5.
    // Remainder {15, 5} carries into the next bucket.
    EXPECT_DOUBLE_EQ(dif.update(60.0, 11.0), 0.5);
    EXPECT_EQ(dif.history().size(), 1u);
}

TEST(Vpin, ResetClearsState) {
    features::DepthImbalanceFlow dif(100.0, 1);
    (void)dif.update(200.0, 10.0);
    ASSERT_FALSE(dif.history().empty());
    dif.reset();
    EXPECT_TRUE(dif.history().empty());
    EXPECT_TRUE(std::isnan(dif.value()));
}

TEST(Vpin, PercentileMatchesStatsHelper) {
    features::DepthImbalanceFlow dif(100.0, 2);
    double mid = 10.0;
    for (int i = 0; i < 50; ++i) {
        mid += (i % 3 == 0) ? 0.1 : -0.05;
        (void)dif.update(100.0, mid);
    }
    ASSERT_FALSE(dif.history().empty());
    EXPECT_DOUBLE_EQ(dif.percentile(90.0),
                     stats::percentile(dif.history(), 90.0));
}

// ── stats helpers ───────────────────────────────────────────────────────────

TEST(Stats, MeanHandComputed) {
    EXPECT_DOUBLE_EQ(stats::mean({1.0, 2.0, 3.0, 4.0}), 2.5);
    EXPECT_DOUBLE_EQ(stats::mean({}), 0.0);
}

TEST(Stats, StddevHandComputed) {
    // Population stddev of {2, 4, 4, 4, 5, 5, 7, 9}: mean = 5, var = 4, σ = 2.
    EXPECT_DOUBLE_EQ(stats::stddev({2, 4, 4, 4, 5, 5, 7, 9}), 2.0);
    EXPECT_DOUBLE_EQ(stats::stddev({42.0}), 0.0);
}

TEST(Stats, PercentileHandComputed) {
    const std::vector<double> v{5.0, 1.0, 3.0, 2.0, 4.0};  // unsorted on purpose
    EXPECT_DOUBLE_EQ(stats::percentile(v, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(stats::percentile(v, 50.0), 3.0);
    EXPECT_DOUBLE_EQ(stats::percentile(v, 100.0), 5.0);
    // Interpolated: idx = 0.25·4 = 1 → exactly the 2nd element.
    EXPECT_DOUBLE_EQ(stats::percentile(v, 25.0), 2.0);
    // idx = 0.625·4 = 2.5 → halfway between 3 and 4.
    EXPECT_DOUBLE_EQ(stats::percentile(v, 62.5), 3.5);
}

TEST(Stats, RmseHandComputed) {
    // Errors {1, −1} → RMSE = sqrt((1+1)/2) = 1.
    EXPECT_DOUBLE_EQ(stats::rmse({2.0, 3.0}, {1.0, 4.0}), 1.0);
    // Identical series → 0.
    EXPECT_DOUBLE_EQ(stats::rmse({1.0, 2.0}, {1.0, 2.0}), 0.0);
}

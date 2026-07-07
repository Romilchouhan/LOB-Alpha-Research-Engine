#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// OFI — Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014)
// ─────────────────────────────────────────────────────────────────────────────
//
// Level-1 OFI per event, computed from two consecutive BBO snapshots:
//
//   e_n =  I(P_b,n ≥ P_b,n−1) · q_b,n   −  I(P_b,n ≤ P_b,n−1) · q_b,n−1
//        − I(P_a,n ≤ P_a,n−1) · q_a,n   +  I(P_a,n ≥ P_a,n−1) · q_a,n−1
//
// Sign intuition:
//   bid price up            → + new bid size        (buy pressure)
//   bid price down          → − old bid size        (bid pulled / hit)
//   bid size up, price flat → + Δ bid size
//   ask size up, price flat → − Δ ask size          (sell pressure)
//
// Also maintains a rolling sum of e_n over a strictly BACKWARD-looking
// window of the last `window` events (the current event included, nothing
// from the future). The first update (no previous snapshot) yields 0.
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/price_level.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>

namespace features {

class OrderFlowImbalance {
public:
    /// @param window  Rolling-sum window length in events (clamped to ≥ 1).
    explicit OrderFlowImbalance(std::size_t window = 50) noexcept
        : window_(std::max<std::size_t>(window, 1)) {}

    /// Feed the next snapshot; returns the per-event OFI e_n.
    double update(const lob::LOBSnapshot& snap) {
        return update(snap.bids[0].price, snap.bids[0].volume,
                      snap.asks[0].price, snap.asks[0].volume);
    }

    /// Feed the next BBO directly; returns the per-event OFI e_n.
    double update(double bid_px, double bid_vol,
                  double ask_px, double ask_vol) {
        double e = 0.0;
        if (has_prev_) {
            if (bid_px >= prev_bid_px_) e += bid_vol;        //  I(Pb ≥) · qb,n
            if (bid_px <= prev_bid_px_) e -= prev_bid_vol_;  // −I(Pb ≤) · qb,n−1
            if (ask_px <= prev_ask_px_) e -= ask_vol;        // −I(Pa ≤) · qa,n
            if (ask_px >= prev_ask_px_) e += prev_ask_vol_;  //  I(Pa ≥) · qa,n−1
        }
        prev_bid_px_  = bid_px;
        prev_bid_vol_ = bid_vol;
        prev_ask_px_  = ask_px;
        prev_ask_vol_ = ask_vol;
        has_prev_     = true;

        // Backward-looking rolling sum (current event included).
        window_buf_.push_back(e);
        rolling_sum_ += e;
        if (window_buf_.size() > window_) {
            rolling_sum_ -= window_buf_.front();
            window_buf_.pop_front();
        }
        last_ = e;
        return e;
    }

    /// Most recent per-event OFI.
    [[nodiscard]] double value() const noexcept { return last_; }

    /// Rolling sum of OFI over the last `window()` events.
    [[nodiscard]] double rolling_sum() const noexcept { return rolling_sum_; }

    [[nodiscard]] std::size_t window() const noexcept { return window_; }

    void reset() noexcept {
        has_prev_ = false;
        last_ = 0.0;
        rolling_sum_ = 0.0;
        window_buf_.clear();
    }

private:
    std::size_t window_;

    bool   has_prev_     = false;
    double prev_bid_px_  = 0.0;
    double prev_bid_vol_ = 0.0;
    double prev_ask_px_  = 0.0;
    double prev_ask_vol_ = 0.0;

    double             last_        = 0.0;
    double             rolling_sum_ = 0.0;
    std::deque<double> window_buf_;
};

} // namespace features

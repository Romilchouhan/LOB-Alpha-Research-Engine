#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// SimulatedTrader — Passive Fill + VPIN Toxicity Abort
// ─────────────────────────────────────────────────────────────────────────────

#include "features/micro_price.hpp"
#include "features/obi.hpp"
#include "features/vpin.hpp"
#include "lob/order_book.hpp"
#include <cstdint>
#include <vector>

namespace trading {

/// Strategy decision taken on a tick.
enum class Action : std::uint8_t {
    Hold,          ///< No signal — do nothing.
    PassiveBuy,    ///< Passive fill at the best bid.
    PassiveSell,   ///< Passive fill at the best ask.
    Abort          ///< VPIN toxicity abort (position flattened if any).
};

/// Stable string form of an Action (for logs / CSV export).
[[nodiscard]] constexpr const char* to_string(Action a) noexcept {
    switch (a) {
        case Action::Hold:        return "HOLD";
        case Action::PassiveBuy:  return "PASSIVE_BUY";
        case Action::PassiveSell: return "PASSIVE_SELL";
        case Action::Abort:       return "ABORT";
    }
    return "UNKNOWN";
}

/// Entry in the trade log (one record per tick).
struct TradeRecord {
    std::uint64_t tick;
    double        price;
    int           side;      // +1 buy, −1 sell, 0 none
    double        pnl;       // total (realised + unrealised) P&L after action
    double        position;  // net position after action
    double        vpin;
    Action        action;
};

/// Configuration knobs for the strategy.
struct TraderConfig {
    double spread_offset      = 0.0001;   // micro_price threshold over mid + offset
    double vpin_toxicity_pct  = 90.0;     // percentile threshold for abort
    double position_limit     = 100.0;    // max absolute position
    double vpin_bucket_vol    = 1000.0;
    std::size_t vpin_window   = 50;
};

class SimulatedTrader {
public:
    explicit SimulatedTrader(const TraderConfig& cfg = {});

    /// Process one LOB snapshot.  Returns the action taken.
    [[nodiscard]] Action on_tick(const lob::LimitOrderBook& book);

    // ── Getters ──────────────────────────────────────────────────────────
    [[nodiscard]] double position()       const noexcept { return position_; }
    [[nodiscard]] double realised_pnl()   const noexcept { return realised_pnl_; }
    [[nodiscard]] double unrealised_pnl(double mark) const noexcept;
    [[nodiscard]] double total_pnl(double mark) const noexcept;
    [[nodiscard]] double inventory_risk() const noexcept;
    [[nodiscard]] std::size_t trade_count() const noexcept { return trades_.size(); }

    [[nodiscard]] const std::vector<TradeRecord>& trades()    const noexcept { return trades_; }
    [[nodiscard]] const std::vector<double>&      pnl_curve() const noexcept { return pnl_curve_; }

    [[nodiscard]] features::VPIN&       vpin_engine() noexcept { return vpin_; }
    [[nodiscard]] const features::VPIN& vpin_engine() const noexcept { return vpin_; }

private:
    TraderConfig          cfg_;
    features::VPIN        vpin_;
    double                position_     = 0.0;
    double                realised_pnl_ = 0.0;
    double                avg_cost_     = 0.0;
    std::vector<TradeRecord> trades_;
    std::vector<double>      pnl_curve_;
};

} // namespace trading

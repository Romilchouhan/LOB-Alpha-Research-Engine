// ─────────────────────────────────────────────────────────────────────────────
// SimulatedTrader — Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include "trading/simulated_trader.hpp"

#include <cmath>

namespace trading {

SimulatedTrader::SimulatedTrader(const TraderConfig& cfg)
    : cfg_(cfg)
    , vpin_(cfg.vpin_bucket_vol, cfg.vpin_window)
{}

Action SimulatedTrader::on_tick(const lob::LimitOrderBook& book) {
    const double mid   = book.mid_price();
    const double micro = features::MicroPrice::compute(book);

    // Feed the VPIN engine — use BBO volume as a proxy for traded volume
    // (FI-2010 is snapshot data; no trade-by-trade feed available).
    const double tick_vol = book.best_ask().volume + book.best_bid().volume;
    const double vpin_val = vpin_.update(tick_vol, mid);

    const std::uint64_t tick = book.tick_count();

    // ── Decision logic ──────────────────────────────────────────────────

    // 1. Check toxicity: ABORT if VPIN exceeds the configured percentile.
    if (vpin_.is_toxic(cfg_.vpin_toxicity_pct)) {
        // If we have a position, flatten it aggressively (mark-to-market close).
        if (std::abs(position_) > 0.0) {
            realised_pnl_ += position_ * (mid - avg_cost_);
            avg_cost_ = 0.0;
            position_ = 0.0;
        }
        pnl_curve_.push_back(total_pnl(mid));
        trades_.push_back({tick, mid, 0, total_pnl(mid), position_, vpin_val,
                           Action::Abort});
        return Action::Abort;
    }

    // 2. Passive Buy: buy when micro_price > mid + spread_offset
    //    (indicates the fair price is above the mid, so our limit
    //    buy at the bid is likely to get filled at a good price).
    if (micro > mid + cfg_.spread_offset && position_ < cfg_.position_limit) {
        // Simulate passive fill at the best bid
        const double fill_px = book.best_bid().price;
        const double old_pos = position_;
        position_ += 1.0;

        // Update average cost
        if (old_pos >= 0.0) {
            avg_cost_ = (avg_cost_ * old_pos + fill_px) / position_;
        } else {
            // Closing a short: realise P&L on the closed portion
            realised_pnl_ += 1.0 * (avg_cost_ - fill_px);
            if (position_ != 0.0)
                avg_cost_ = fill_px;
        }

        pnl_curve_.push_back(total_pnl(mid));
        trades_.push_back({tick, fill_px, +1, total_pnl(mid), position_,
                           vpin_val, Action::PassiveBuy});
        return Action::PassiveBuy;
    }

    // 3. Passive Sell: sell when micro_price < mid - spread_offset
    if (micro < mid - cfg_.spread_offset && position_ > -cfg_.position_limit) {
        const double fill_px = book.best_ask().price;
        const double old_pos = position_;
        position_ -= 1.0;

        if (old_pos <= 0.0) {
            avg_cost_ = (avg_cost_ * std::abs(old_pos) + fill_px)
                        / std::abs(position_);
        } else {
            realised_pnl_ += 1.0 * (fill_px - avg_cost_);
            if (position_ != 0.0)
                avg_cost_ = fill_px;
        }

        pnl_curve_.push_back(total_pnl(mid));
        trades_.push_back({tick, fill_px, -1, total_pnl(mid), position_,
                           vpin_val, Action::PassiveSell});
        return Action::PassiveSell;
    }

    // 4. HOLD — no signal
    pnl_curve_.push_back(total_pnl(mid));
    trades_.push_back({tick, mid, 0, total_pnl(mid), position_, vpin_val,
                       Action::Hold});
    return Action::Hold;
}

double SimulatedTrader::unrealised_pnl(double mark) const noexcept {
    if (position_ == 0.0) return 0.0;
    if (position_ > 0.0)
        return position_ * (mark - avg_cost_);
    else
        return std::abs(position_) * (avg_cost_ - mark);
}

double SimulatedTrader::total_pnl(double mark) const noexcept {
    return realised_pnl_ + unrealised_pnl(mark);
}

double SimulatedTrader::inventory_risk() const noexcept {
    // Simple squared-position penalty as risk measure
    return position_ * position_;
}

} // namespace trading

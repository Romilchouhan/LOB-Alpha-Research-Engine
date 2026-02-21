#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// PriceLevel — atomic unit of the Limit Order Book
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>

namespace lob {

/// A single price level (one row on one side of the book).
struct PriceLevel {
    double price  = 0.0;   // limit price
    double volume = 0.0;   // aggregate resting volume at this level

    [[nodiscard]] constexpr bool empty() const noexcept { return volume <= 0.0; }
};

/// Raw snapshot row count for FI-2010 (10 levels × 2 sides × 2 fields).
inline constexpr std::size_t kFI2010Columns = 40;

/// Number of depth levels on each side.
inline constexpr int kMaxDepth = 10;

/// Packed LOB snapshot — exactly what the binary file stores.
struct alignas(64) LOBSnapshot {
    PriceLevel asks[kMaxDepth];   // asks[0] = best ask  (lowest)
    PriceLevel bids[kMaxDepth];   // bids[0] = best bid  (highest)
};

static_assert(sizeof(PriceLevel) == 16);
static_assert(sizeof(LOBSnapshot) == 320);

} // namespace lob

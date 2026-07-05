// ─────────────────────────────────────────────────────────────────────────────
// FI-2010 CSV & Binary Parser — Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/fi2010_parser.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace lob {

// ── CSV parsing ─────────────────────────────────────────────────────────────

static bool is_header_line(const std::string& line) {
    if (line.empty()) return true;
    const char c = line.front();
    return c == '#' || std::isalpha(static_cast<unsigned char>(c));
}

static double fast_parse_double(const char* begin, const char* /*end*/) {
    return std::strtod(begin, nullptr);
}

std::vector<LOBSnapshot> parse_fi2010_csv(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in.is_open())
        throw std::runtime_error("Cannot open CSV: " + file.string());

    std::vector<LOBSnapshot> snapshots;
    snapshots.reserve(150'000);  // FI-2010 has ~150 k rows per day

    std::string line;
    while (std::getline(in, line)) {
        if (is_header_line(line)) continue;

        // Tokenise on comma
        double vals[kFI2010Columns]{};
        std::size_t col = 0;
        const char* p = line.data();
        const char* end = p + line.size();

        while (p < end && col < kFI2010Columns) {
            const char* comma = static_cast<const char*>(
                std::memchr(p, ',', static_cast<std::size_t>(end - p)));
            if (!comma) comma = end;
            vals[col++] = fast_parse_double(p, comma);
            p = comma + 1;
        }

        if (col < kFI2010Columns) continue;  // malformed row

        // FI-2010 canonical column order:
        //   0-9  : ask_price_1 … ask_price_10
        //  10-19 : ask_vol_1   … ask_vol_10
        //  20-29 : bid_price_1 … bid_price_10
        //  30-39 : bid_vol_1   … bid_vol_10
        LOBSnapshot snap{};
        for (int i = 0; i < kMaxDepth; ++i) {
            snap.asks[i].price  = vals[i];
            snap.asks[i].volume = vals[i + 10];
            snap.bids[i].price  = vals[i + 20];
            snap.bids[i].volume = vals[i + 30];
        }
        snapshots.push_back(snap);
    }

    std::cout << "[Parser] Loaded " << snapshots.size()
              << " snapshots from CSV\n";
    return snapshots;
}

// ── Binary I/O ──────────────────────────────────────────────────────────────

std::vector<LOBSnapshot> read_binary(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in.is_open())
        throw std::runtime_error("Cannot open binary: " + file.string());

    const auto size = in.tellg();
    if (size % sizeof(LOBSnapshot) != 0)
        throw std::runtime_error("Binary file size not aligned to LOBSnapshot");

    const std::size_t count = static_cast<std::size_t>(size) / sizeof(LOBSnapshot);
    std::vector<LOBSnapshot> snaps(count);

    in.seekg(0);
    in.read(reinterpret_cast<char*>(snaps.data()),
            static_cast<std::streamsize>(size));

    std::cout << "[Parser] Loaded " << count << " snapshots from binary\n";
    return snaps;
}

// ── Synthetic data generator ────────────────────────────────────────────────

std::vector<LOBSnapshot> generate_synthetic(std::size_t n) {
    std::mt19937 rng(42);
    std::normal_distribution<double> price_noise(0.0, 0.05);
    std::uniform_real_distribution<double> vol_dist(10.0, 500.0);

    std::vector<LOBSnapshot> snaps;
    snaps.reserve(n);

    double base_price = 100.0;

    for (std::size_t t = 0; t < n; ++t) {
        base_price += price_noise(rng);
        const double spread = 0.02 + std::abs(price_noise(rng)) * 0.01;

        LOBSnapshot snap{};
        for (int i = 0; i < kMaxDepth; ++i) {
            snap.asks[i].price  = base_price + spread * 0.5 + 0.01 * i;
            snap.asks[i].volume = vol_dist(rng);
            snap.bids[i].price  = base_price - spread * 0.5 - 0.01 * i;
            snap.bids[i].volume = vol_dist(rng);
        }
        snaps.push_back(snap);
    }

    std::cout << "[Synthetic] Generated " << n << " snapshots\n";
    return snaps;
}

} // namespace lob

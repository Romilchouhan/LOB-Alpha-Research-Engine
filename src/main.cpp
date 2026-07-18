// ─────────────────────────────────────────────────────────────────────────────
// LOB Alpha Research Engine — Main entry point
// ─────────────────────────────────────────────────────────────────────────────
//
// Usage:
//   lob_engine --file data/lob.bin            # pre-processed binary
//   lob_engine --csv  data/fi2010.csv         # raw FI-2010 CSV
//   lob_engine --synthetic                    # 500-tick synthetic book
//   lob_engine --file data/lob.bin --plot
// ─────────────────────────────────────────────────────────────────────────────

#include "lob/fi2010_parser.hpp"
#include "lob/order_book.hpp"
#include "features/accel.hpp"
#include "features/book_slope.hpp"
#include "features/micro_price.hpp"
#include "features/obi.hpp"
#include "features/ofi.hpp"
#include "features/queue_imbalance.hpp"
#include "features/realized_vol.hpp"
#include "features/depth_imbalance_flow.hpp"
#include "stats/rmse.hpp"
#include "trading/simulated_trader.hpp"

#ifdef LOB_HAS_PARQUET
#include "io/parquet_writer.hpp"
#include <optional>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Forward declarations from visualization.cpp
namespace viz {
void plot_pnl(const std::vector<double>&, const std::string&);
void plot_micro_vs_mid(const std::vector<double>&, const std::vector<double>&,
                       const std::string&);
void plot_dif(const std::vector<double>&, double, const std::string&);
}

// ── CLI argument parser ─────────────────────────────────────────────────────

struct Args {
    std::string csv_path;
    std::string bin_path;
    std::string dump_csv;        // --dump-csv <path>
    std::string emit_parquet;    // --emit-parquet <path>
    bool        synthetic     = false;
    bool        plot          = false;
    double      spread_offset = 0.0001;
    double      dif_bucket   = 1000.0;
    std::size_t dif_window   = 50;
    std::size_t synthetic_n   = 500;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            a.csv_path = argv[++i];
        else if (std::strcmp(argv[i], "--file") == 0 && i + 1 < argc)
            a.bin_path = argv[++i];
        else if (std::strcmp(argv[i], "--synthetic") == 0)
            a.synthetic = true;
        else if (std::strcmp(argv[i], "--plot") == 0)
            a.plot = true;
        else if (std::strcmp(argv[i], "--dump-csv") == 0 && i + 1 < argc)
            a.dump_csv = argv[++i];
        else if (std::strcmp(argv[i], "--emit-parquet") == 0 && i + 1 < argc)
            a.emit_parquet = argv[++i];
        else if (std::strcmp(argv[i], "--spread-offset") == 0 && i + 1 < argc)
            a.spread_offset = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--dif-bucket") == 0 && i + 1 < argc)
            a.dif_bucket = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--dif-window") == 0 && i + 1 < argc)
            a.dif_window = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (std::strcmp(argv[i], "--synthetic-n") == 0 && i + 1 < argc)
            a.synthetic_n = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout <<
                "LOB Alpha Research Engine — FI-2010\n\n"
                "  --csv  <path>       Load FI-2010 CSV file\n"
                "  --file <path>       Load pre-processed binary file\n"
                "  --synthetic         Generate synthetic LOB data\n"
                "  --synthetic-n <n>   Number of synthetic ticks (default 500)\n"
                "  --dump-csv <path>   Export tick data to CSV for plotting\n"
                "  --emit-parquet <p>  Export full feature matrix to Parquet\n"
                "  --plot              Show Matplotplusplus charts (if compiled)\n"
                "  --spread-offset <d> Micro-price spread offset (default 0.0001)\n"
                "  --dif-bucket <d>   DepthImbalanceFlow bucket volume (default 1000)\n"
                "  --dif-window <n>   DepthImbalanceFlow rolling window buckets (default 50)\n"
                "  --help              Show this message\n";
            std::exit(0);
        }
    }
    return a;
}

// ── RMSE research task ──────────────────────────────────────────────────────

static void run_rmse_analysis(const std::vector<double>& micro_series,
                              const std::vector<double>& mid_series,
                              const std::vector<int>& horizons) {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Micro-Price Predictive Accuracy (RMSE)         ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣\n";
    std::cout << "║  Horizon (ticks)  │  RMSE              ║\n";
    std::cout << "╠═══════════════════╪═════════════════════╣\n";

    for (int h : horizons) {
        if (static_cast<std::size_t>(h) >= mid_series.size()) {
            std::cout << "║  " << std::setw(16) << h
                      << " │  (insufficient data)      ║\n";
            continue;
        }

        // predicted[t] = micro_series[t]
        // actual[t]    = mid_series[t + h]
        const std::size_t n = mid_series.size() - static_cast<std::size_t>(h);
        std::vector<double> predicted(micro_series.begin(),
                                      micro_series.begin() + static_cast<long>(n));
        std::vector<double> actual(mid_series.begin() + h,
                                   mid_series.end());

        const double r = stats::rmse(predicted, actual);
        std::cout << "║  " << std::setw(16) << h
                  << " │  " << std::setw(18) << std::fixed
                  << std::setprecision(8) << r << " ║\n";
    }
    std::cout << "╚═══════════════════╧═════════════════════╝\n";
}

// ── Engine body (wrapped by main's try/catch) ───────────────────────────────

static int run_engine(const Args& args) {
    // ── 1. Load data ────────────────────────────────────────────────────
    std::vector<lob::LOBSnapshot> snapshots;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!args.bin_path.empty()) {
        snapshots = lob::read_binary(args.bin_path);
    } else if (!args.csv_path.empty()) {
        snapshots = lob::parse_fi2010_csv(args.csv_path);
    } else {
        std::cout << "[Engine] No input file specified — using synthetic data\n";
        snapshots = lob::generate_synthetic(args.synthetic_n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Engine] Data loaded in " << std::fixed << std::setprecision(2)
              << load_ms << " ms  (" << snapshots.size() << " snapshots)\n";

    if (snapshots.empty()) {
        std::cerr << "[Engine] No data to process.\n";
        return 1;
    }

    // ── 2. Set up components ────────────────────────────────────────────
    lob::LimitOrderBook book;

    trading::TraderConfig tcfg;
    tcfg.spread_offset     = args.spread_offset;
    tcfg.dif_bucket_vol   = args.dif_bucket;
    tcfg.dif_window       = args.dif_window;
    trading::SimulatedTrader trader(tcfg);

    // Time-series accumulators
    std::vector<double> micro_series, mid_series, obi_series;
    micro_series.reserve(snapshots.size());
    mid_series.reserve(snapshots.size());
    obi_series.reserve(snapshots.size());

    // P2 feature engines (all strictly backward-looking)
    features::OrderFlowImbalance ofi_engine(50);
    features::RealizedVolatility rv50_engine(50);
    features::RealizedVolatility rv200_engine(200);
    features::MidAcceleration    accel_engine;

#ifdef LOB_HAS_PARQUET
    std::optional<io::ParquetWriter> parquet;
    if (!args.emit_parquet.empty())
        parquet.emplace(args.emit_parquet);
#else
    if (!args.emit_parquet.empty())
        std::cerr << "[Engine] --emit-parquet ignored: built without Arrow/Parquet "
                     "(configure with -DLOB_ENABLE_PARQUET=ON and install "
                     "apache-arrow)\n";
#endif

    // ── 3. Main tick loop ───────────────────────────────────────────────
    std::size_t fill_count = 0, abort_count = 0, hold_count = 0;

    auto t2 = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < snapshots.size(); ++i) {
        book.update(snapshots[i]);

        const double micro = features::MicroPrice::compute(book);
        const double mid   = book.mid_price();
        const double obi   = features::OrderBookImbalance::compute(book);

        micro_series.push_back(micro);
        mid_series.push_back(mid);
        obi_series.push_back(obi);

        // ── P2 feature updates (cheap, O(1) each) ───────────────────
        const double ofi   = ofi_engine.update(snapshots[i]);
        const double rv50  = rv50_engine.update(mid);
        const double rv200 = rv200_engine.update(mid);
        const double accel = accel_engine.update(mid);

        // ── Trader decision ─────────────────────────────────────────
        switch (trader.on_tick(book)) {
            case trading::Action::PassiveBuy:
            case trading::Action::PassiveSell: ++fill_count;  break;
            case trading::Action::Abort:       ++abort_count; break;
            case trading::Action::Hold:        ++hold_count;  break;
        }

#ifdef LOB_HAS_PARQUET
        if (parquet) {
            const auto slope = features::BookSlope::compute(book);
            io::FeatureRow row;
            row.tick        = static_cast<std::int64_t>(i);
            row.mid         = mid;
            row.spread      = book.spread();
            row.micro_price = micro;
            row.obi         = obi;
            row.ofi         = ofi;
            row.ofi_rolling = ofi_engine.rolling_sum();
            row.bid_slope   = slope.bid_slope;
            row.ask_slope   = slope.ask_slope;
            row.qimb        = features::QueueImbalance::compute(book);
            row.rv_50       = rv50;
            row.rv_200      = rv200;
            row.accel       = accel;
            row.dif        = trader.dif_engine().value();
            parquet->append(row);
        }
#else
        (void)ofi; (void)rv50; (void)rv200; (void)accel;
#endif
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    const double loop_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ── 4. Research output ──────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║          LOB Alpha Research Engine               ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣\n";
    std::cout << "║  Ticks processed : " << std::setw(10) << snapshots.size()
              << "                    ║\n";
    std::cout << "║  Processing time : " << std::setw(10) << std::fixed
              << std::setprecision(2) << loop_ms << " ms"
              << "               ║\n";
    std::cout << "║  Throughput      : " << std::setw(10) << std::fixed
              << std::setprecision(0)
              << (snapshots.size() / (loop_ms / 1000.0)) << " ticks/s"
              << "          ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣\n";
    std::cout << "║  FILLS  : " << std::setw(8) << fill_count
              << "    ABORTS : " << std::setw(8) << abort_count
              << "    ║\n";
    std::cout << "║  HOLDS  : " << std::setw(8) << hold_count
              << "    TRADES : " << std::setw(8) << trader.trade_count()
              << "    ║\n";
    std::cout << "║  Position: " << std::setw(+10) << std::fixed
              << std::setprecision(2) << trader.position()
              << "                         ║\n";
    std::cout << "║  PnL     : " << std::setw(+10) << std::fixed
              << std::setprecision(4)
              << trader.total_pnl(mid_series.back())
              << "                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    // ── 5. CSV export ──────────────────────────────────────────────────
    if (!args.dump_csv.empty()) {
        std::ofstream csv(args.dump_csv);
        csv << "tick,mid_price,micro_price,obi,spread,pnl,position,depth_imb_flow,action\n";
        const auto& trades = trader.trades();
        for (std::size_t i = 0; i < snapshots.size(); ++i) {
            csv << i
                << "," << std::fixed << std::setprecision(8) << mid_series[i]
                << "," << micro_series[i]
                << "," << obi_series[i]
                << "," << (snapshots[i].asks[0].price - snapshots[i].bids[0].price)
                << "," << (i < trades.size() ? trades[i].pnl : 0.0)
                << "," << (i < trades.size() ? trades[i].position : 0.0)
                << "," << (i < trades.size() ? trades[i].dif : 0.0)
                << "," << (i < trades.size() ? trading::to_string(trades[i].action)
                                             : "")
                << "\n";
        }
        std::cout << "[Engine] Exported " << snapshots.size()
                  << " ticks to " << args.dump_csv << "\n";
    }

    // ── 5b. Parquet feature-matrix export ──────────────────────────────
#ifdef LOB_HAS_PARQUET
    if (parquet) {
        parquet->finalize();
        std::cout << "[Engine] Wrote " << parquet->rows()
                  << "-row feature matrix to " << parquet->path().string()
                  << "\n";
    }
#endif

    // ── 6. RMSE analysis ────────────────────────────────────────────────
    run_rmse_analysis(micro_series, mid_series, {10, 50, 100});

    // ── 7. DepthImbalanceFlow summary ─────────────────────────────────────────────────
    const auto& dif_hist = trader.dif_engine().history();
    if (!dif_hist.empty()) {
        std::cout << "\n[DepthImbalanceFlow] Observations: " << dif_hist.size()
                  << "  |  Mean: " << std::fixed << std::setprecision(6)
                  << stats::mean(dif_hist)
                  << "  |  90th pct: "
                  << trader.dif_engine().percentile(90.0)
                  << "  |  Max: "
                  << *std::max_element(dif_hist.begin(), dif_hist.end())
                  << "\n";
    }

    // ── 8. Visualization (if enabled) ───────────────────────────────────
    if (args.plot) {
        viz::plot_pnl(trader.pnl_curve(), "SimulatedTrader PnL");
        viz::plot_micro_vs_mid(micro_series, mid_series,
                               "Micro-Price vs Mid-Price");
        if (!dif_hist.empty())
            viz::plot_dif(dif_hist,
                           trader.dif_engine().percentile(90.0),
                           "DepthImbalanceFlow Time-Series");
    }

    return 0;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        return run_engine(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Fatal error: " << e.what() << "\n";
        return 1;
    }
}

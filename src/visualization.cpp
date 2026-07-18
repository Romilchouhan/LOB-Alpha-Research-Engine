// ─────────────────────────────────────────────────────────────────────────────
// Visualization — Matplotplusplus wrapper
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <string>
#include <vector>

#if defined(LOB_HAS_MATPLOT) && LOB_HAS_MATPLOT
#include <matplot/matplot.h>
#endif

namespace viz {

void plot_pnl(const std::vector<double>& pnl_curve,
              const std::string& title = "PnL Curve") {
#if defined(LOB_HAS_MATPLOT) && LOB_HAS_MATPLOT
    using namespace matplot;
    auto f = figure(true);
    f->size(1200, 600);
    plot(pnl_curve)->line_width(1.5);
    matplot::title(title);
    xlabel("Tick");
    ylabel("PnL");
    grid(true);
    show();
#else
    (void)title;
    std::cout << "[Viz] Matplotplusplus not enabled. PnL summary:\n"
              << "      Points  : " << pnl_curve.size() << "\n"
              << "      Start   : " << (pnl_curve.empty() ? 0.0 : pnl_curve.front()) << "\n"
              << "      End     : " << (pnl_curve.empty() ? 0.0 : pnl_curve.back()) << "\n";
#endif
}

void plot_micro_vs_mid(const std::vector<double>& micro,
                       const std::vector<double>& mid,
                       const std::string& title = "Micro-Price vs Mid-Price") {
#if defined(LOB_HAS_MATPLOT) && LOB_HAS_MATPLOT
    using namespace matplot;
    auto f = figure(true);
    f->size(1200, 600);
    hold(on);
    auto p1 = plot(mid);
    p1->display_name("Mid-Price");
    p1->line_width(1.2);
    auto p2 = plot(micro);
    p2->display_name("Micro-Price");
    p2->line_width(1.2);
    hold(off);
    matplot::title(title);
    xlabel("Tick");
    ylabel("Price");
    legend();
    grid(true);
    show();
#else
    (void)title;
    std::cout << "[Viz] Matplotplusplus not enabled. Series lengths: "
              << "micro=" << micro.size() << ", mid=" << mid.size() << "\n";
#endif
}

void plot_dif(const std::vector<double>& dif_series,
               double threshold,
               const std::string& title = "DepthImbalanceFlow Time-Series") {
#if defined(LOB_HAS_MATPLOT) && LOB_HAS_MATPLOT
    using namespace matplot;
    auto f = figure(true);
    f->size(1200, 600);
    hold(on);
    auto p1 = plot(dif_series);
    p1->display_name("DepthImbalanceFlow");
    p1->line_width(1.2);
    auto p2 = plot(std::vector<double>(dif_series.size(), threshold));
    p2->display_name("90th Percentile");
    p2->line_width(1.0).color("r").line_style("--");
    hold(off);
    matplot::title(title);
    xlabel("Bucket");
    ylabel("DepthImbalanceFlow");
    legend();
    grid(true);
    show();
#else
    (void)threshold;
    (void)title;
    std::cout << "[Viz] Matplotplusplus not enabled. DepthImbalanceFlow points: "
              << dif_series.size() << "\n";
#endif
}

} // namespace viz

// ─────────────────────────────────────────────────────────────────────────────
// ParquetWriter — implementation (Apache Arrow / Parquet)
// ─────────────────────────────────────────────────────────────────────────────

#include "io/parquet_writer.hpp"

// Workaround: Apple's SDK ships std::bit_width but omits the __cpp_lib_bitops
// feature-test macro, which makes arrow/util/bit_util.h fall back to
// std::log2p1 — a pre-C++20-final name that never shipped in libc++. Assert
// bit-ops support before pulling in Arrow so it takes the std::bit_width path.
#include <bit>
#if defined(__apple_build_version__) && !defined(__cpp_lib_bitops)
#define __cpp_lib_bitops 201907L
#endif

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace io {

namespace {

[[noreturn]] void throw_arrow(const std::string& what, const arrow::Status& st) {
    throw std::runtime_error("[ParquetWriter] " + what + ": " + st.ToString());
}

std::shared_ptr<arrow::Array> make_double_array(const std::vector<double>& v) {
    arrow::DoubleBuilder builder;
    if (auto st = builder.AppendValues(v); !st.ok())
        throw_arrow("append doubles", st);
    std::shared_ptr<arrow::Array> out;
    if (auto st = builder.Finish(&out); !st.ok())
        throw_arrow("finish double array", st);
    return out;
}

std::shared_ptr<arrow::Array> make_int64_array(const std::vector<std::int64_t>& v) {
    arrow::Int64Builder builder;
    if (auto st = builder.AppendValues(v); !st.ok())
        throw_arrow("append int64s", st);
    std::shared_ptr<arrow::Array> out;
    if (auto st = builder.Finish(&out); !st.ok())
        throw_arrow("finish int64 array", st);
    return out;
}

} // namespace

ParquetWriter::ParquetWriter(std::filesystem::path path)
    : path_(std::move(path)) {}

void ParquetWriter::append(const FeatureRow& row) {
    tick_.push_back(row.tick);
    mid_.push_back(row.mid);
    spread_.push_back(row.spread);
    micro_price_.push_back(row.micro_price);
    obi_.push_back(row.obi);
    ofi_.push_back(row.ofi);
    ofi_rolling_.push_back(row.ofi_rolling);
    bid_slope_.push_back(row.bid_slope);
    ask_slope_.push_back(row.ask_slope);
    for (int i = 0; i < lob::kMaxDepth; ++i)
        qimb_[static_cast<std::size_t>(i)].push_back(
            row.qimb[static_cast<std::size_t>(i)]);
    rv_50_.push_back(row.rv_50);
    rv_200_.push_back(row.rv_200);
    accel_.push_back(row.accel);
    vpin_.push_back(row.vpin);
}

void ParquetWriter::finalize() {
    if (finalized_) return;

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(23);
    arrays.reserve(23);

    const auto add_dbl = [&](const std::string& name,
                             const std::vector<double>& col) {
        fields.push_back(arrow::field(name, arrow::float64()));
        arrays.push_back(make_double_array(col));
    };

    fields.push_back(arrow::field("tick", arrow::int64()));
    arrays.push_back(make_int64_array(tick_));

    add_dbl("mid", mid_);
    add_dbl("spread", spread_);
    add_dbl("micro_price", micro_price_);
    add_dbl("obi", obi_);
    add_dbl("ofi", ofi_);
    add_dbl("ofi_rolling", ofi_rolling_);
    add_dbl("bid_slope", bid_slope_);
    add_dbl("ask_slope", ask_slope_);
    for (int i = 0; i < lob::kMaxDepth; ++i)
        add_dbl("qimb_" + std::to_string(i + 1),
                qimb_[static_cast<std::size_t>(i)]);
    add_dbl("rv_50", rv_50_);
    add_dbl("rv_200", rv_200_);
    add_dbl("accel", accel_);
    add_dbl("vpin", vpin_);

    const auto schema = std::make_shared<arrow::Schema>(fields);
    const auto table  = arrow::Table::Make(schema, arrays,
                                           static_cast<std::int64_t>(rows()));

    auto out_res = arrow::io::FileOutputStream::Open(path_.string());
    if (!out_res.ok())
        throw_arrow("open " + path_.string(), out_res.status());

    if (auto st = parquet::arrow::WriteTable(
            *table, arrow::default_memory_pool(), *out_res,
            /*chunk_size=*/64 * 1024);
        !st.ok())
        throw_arrow("write table", st);

    finalized_ = true;
}

} // namespace io

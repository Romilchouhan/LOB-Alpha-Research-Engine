# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build the C++ engine (Parquet + tests OFF for a slim, fast image;
# the dashboard reads the CSV feature matrix, which carries the full feature set)
# ─────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tests ./tests

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DLOB_ENABLE_PARQUET=OFF \
        -DLOB_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python runtime serving the Streamlit dashboard
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

COPY --from=builder /src/build/lob_engine /usr/local/bin/lob_engine
COPY dashboard ./dashboard
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV LOB_DATA_DIR=/app/data
EXPOSE 8501

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

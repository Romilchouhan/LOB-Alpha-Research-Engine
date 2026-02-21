cmake_minimum_required(VERSION 3.22)
project(LOB_Alpha_Research_Engine_FI2010
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "HFT Micro-Price Research Engine for FI-2010 LOB Data"
)

# ── C++20 ────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Apple-Silicon / ARM64 tuning ─────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m2 -DNDEBUG")
    message(STATUS "[LOB] Targeting Apple Silicon (arm64) — -mcpu=apple-m2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

# ── Options ──────────────────────────────────────────────────────────────────
option(LOB_ENABLE_HAZELCAST "Build with Hazelcast C++ Client support"  OFF)
option(LOB_ENABLE_PLOTTING  "Build with Matplotplusplus support"       OFF)
option(LOB_ENABLE_BOOST     "Use Boost.Accumulators for statistics"    OFF)

# ── FetchContent ─────────────────────────────────────────────────────────────
include(FetchContent)

# Matplotplusplus (optional — heavy build)
if(LOB_ENABLE_PLOTTING)
    FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
        GIT_TAG        v1.2.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(matplotplusplus)
    message(STATUS "[LOB] Matplotplusplus ENABLED")
endif()

# Hazelcast C++ Client (optional)
if(LOB_ENABLE_HAZELCAST)
    FetchContent_Declare(
        hazelcast
        GIT_REPOSITORY https://github.com/hazelcast/hazelcast-cpp-client.git
        GIT_TAG        v5.5.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(hazelcast)
    # Force C++20 on the Hazelcast target to resolve Boost 1.90+ / Apple Clang
    # compatibility (std::decay_t needs proper <type_traits> inclusion).
    if(TARGET hazelcast-cpp-client)
        set_target_properties(hazelcast-cpp-client PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
    endif()
    message(STATUS "[LOB] Hazelcast C++ Client ENABLED")
endif()

# Boost (optional)
if(LOB_ENABLE_BOOST)
    find_package(Boost 1.74 REQUIRED)
    message(STATUS "[LOB] Boost ENABLED — ${Boost_INCLUDE_DIRS}")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(LOB_SOURCES
    src/main.cpp
    src/fi2010_parser.cpp
    src/vpin.cpp
    src/simulated_trader.cpp
    src/hazelcast_store.cpp
    src/visualization.cpp
)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(lob_engine ${LOB_SOURCES})

target_include_directories(lob_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Compile definitions for optional features
if(LOB_ENABLE_HAZELCAST)
    target_compile_definitions(lob_engine PRIVATE LOB_HAS_HAZELCAST=1)
    target_link_libraries(lob_engine PRIVATE hazelcast-cpp-client)
endif()

if(LOB_ENABLE_PLOTTING)
    target_compile_definitions(lob_engine PRIVATE LOB_HAS_MATPLOT=1)
    target_link_libraries(lob_engine PRIVATE matplot)
endif()

if(LOB_ENABLE_BOOST)
    target_compile_definitions(lob_engine PRIVATE LOB_HAS_BOOST=1)
    target_include_directories(lob_engine PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS lob_engine DESTINATION bin)

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════╗")
message(STATUS "║  LOB Alpha Research Engine — FI-2010                       ║")
message(STATUS "║  C++20 · Apple Silicon · HFT Research                      ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Hazelcast  : ${LOB_ENABLE_HAZELCAST}")
message(STATUS "║  Plotting   : ${LOB_ENABLE_PLOTTING}")
message(STATUS "║  Boost      : ${LOB_ENABLE_BOOST}")
message(STATUS "╚══════════════════════════════════════════════════════════════╝")
message(STATUS "")

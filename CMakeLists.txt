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
option(LOB_ENABLE_PLOTTING   "Build with Matplotplusplus support"           OFF)
option(LOB_ENABLE_SANITIZERS "Build with ASan + UBSan instrumentation"      OFF)

# ── Warnings (project targets only — never fetched third-party code) ────────
set(LOB_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)

# ── Sanitizers ───────────────────────────────────────────────────────────────
if(LOB_ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    message(STATUS "[LOB] Sanitizers ENABLED (ASan + UBSan)")
endif()

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

# ── Core library (shared between the engine and the test suite) ─────────────
add_library(lob_core STATIC
    src/fi2010_parser.cpp
    src/vpin.cpp
    src/simulated_trader.cpp
)
target_include_directories(lob_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(lob_core PRIVATE ${LOB_WARNING_FLAGS})

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(lob_engine
    src/main.cpp
    src/visualization.cpp
)
target_link_libraries(lob_engine PRIVATE lob_core)
target_compile_options(lob_engine PRIVATE ${LOB_WARNING_FLAGS})

if(LOB_ENABLE_PLOTTING)
    target_compile_definitions(lob_engine PRIVATE LOB_HAS_MATPLOT=1)
    target_link_libraries(lob_engine PRIVATE matplot)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
# Windows-only, but harmless and documented gtest best practice.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS lob_engine DESTINATION bin)

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════╗")
message(STATUS "║  LOB Alpha Research Engine — FI-2010                       ║")
message(STATUS "║  C++20 · Apple Silicon · HFT Research                      ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Plotting   : ${LOB_ENABLE_PLOTTING}")
message(STATUS "║  Sanitizers : ${LOB_ENABLE_SANITIZERS}")
message(STATUS "╚══════════════════════════════════════════════════════════════╝")
message(STATUS "")

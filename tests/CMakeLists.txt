# ── Test suite ───────────────────────────────────────────────────────────────
add_executable(lob_tests
    test_lob.cpp
    test_features.cpp
)
target_link_libraries(lob_tests PRIVATE lob_core GTest::gtest_main)
target_compile_options(lob_tests PRIVATE ${LOB_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(lob_tests)

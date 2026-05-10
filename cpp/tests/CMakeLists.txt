# ----------------------------------------------------------------------------
# Doctest — same FetchContent + vendored-fallback pattern as vol-surface.
# ----------------------------------------------------------------------------

set(DF_DOCTEST_INCLUDE_DIR "" CACHE PATH
    "Optional path to a vendored doctest header (skip FetchContent if set)")

if(DF_DOCTEST_INCLUDE_DIR)
    add_library(doctest_iface INTERFACE)
    target_include_directories(doctest_iface
        INTERFACE ${DF_DOCTEST_INCLUDE_DIR})
    add_library(doctest::doctest ALIAS doctest_iface)
    message(STATUS "Tests: vendored doctest at ${DF_DOCTEST_INCLUDE_DIR}")
else()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(doctest)
    message(STATUS "Tests: FetchContent doctest v2.4.11")
endif()

add_executable(df_tests
    test_main.cpp
    test_linalg.cpp
    test_pca.cpp
    test_kalman.cpp
    test_rolling.cpp
)
target_link_libraries(df_tests PRIVATE df_kernel doctest::doctest)

if(NOT DF_DOCTEST_INCLUDE_DIR)
    include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
    doctest_discover_tests(df_tests)
else()
    add_test(NAME df_tests COMMAND df_tests)
endif()

cmake_minimum_required(VERSION 3.20)

project(dynamic_factors
    VERSION 0.1.0
    DESCRIPTION "Dynamic factor model for equity stat-arb — PCA + Kalman filtering, C++ kernel + Python orchestration"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global C++ standard.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(DF_BUILD_TESTS    "Build C++ unit tests" ON)
option(DF_BUILD_PYTHON   "Build pybind11 Python module" OFF)
option(DF_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# enable_testing must live at the top level so ctest picks up tests
# registered in subdirectories.
if(DF_BUILD_TESTS)
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(cpp)

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "dynamic_factors ${PROJECT_VERSION} configured")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests:             ${DF_BUILD_TESTS}")
message(STATUS "  Python bindings:   ${DF_BUILD_PYTHON}")
message(STATUS "  Warnings as error: ${DF_WARNINGS_AS_ERRORS}")
message(STATUS "")

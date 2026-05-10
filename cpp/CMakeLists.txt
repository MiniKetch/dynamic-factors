# ----------------------------------------------------------------------------
# Eigen — header-only linear algebra library.
#
# Pulled via FetchContent so users don't have to install it system-wide.
# Eigen is the canonical C++ choice for dense linear algebra: zero runtime
# dependency, expression-template-driven (so multiplications fuse), and
# the SelfAdjointEigenSolver is among the fastest open-source symmetric
# eigendecomposition implementations.
# ----------------------------------------------------------------------------
include(FetchContent)

# CMake 4.x stripped <3.5 compatibility; Eigen still declares 3.5. Floor
# the policy version so configure doesn't error.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
# We don't want Eigen's own targets/install rules — we only need its
# header path. EXCLUDE_FROM_ALL keeps Eigen's CMake from polluting ours.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# ----------------------------------------------------------------------------
# df_kernel — the C++ math library
# ----------------------------------------------------------------------------
add_library(df_kernel STATIC
    src/linalg.cpp
    src/pca.cpp
)

target_include_directories(df_kernel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(df_kernel PUBLIC Eigen3::Eigen)

set_target_properties(df_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ----------------------------------------------------------------------------
# Strict warnings on our own code — keep Eigen's headers warning-quiet.
# ----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(df_kernel PRIVATE /W4 /permissive-)
    if(DF_WARNINGS_AS_ERRORS)
        target_compile_options(df_kernel PRIVATE /WX)
    endif()
else()
    target_compile_options(df_kernel PRIVATE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion
        -Wformat=2
    )
    if(DF_WARNINGS_AS_ERRORS)
        target_compile_options(df_kernel PRIVATE -Werror)
    endif()
endif()

# ----------------------------------------------------------------------------
# Tests / Python bindings (opt-in)
# ----------------------------------------------------------------------------
if(DF_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(DF_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

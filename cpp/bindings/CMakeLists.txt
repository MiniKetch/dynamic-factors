# Python extension module via pybind11.

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_df_kernel MODULE
    python_module.cpp
)

target_link_libraries(_df_kernel PRIVATE df_kernel)

install(TARGETS _df_kernel
        LIBRARY DESTINATION dynamic_factors
        RUNTIME DESTINATION dynamic_factors)

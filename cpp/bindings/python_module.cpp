// dynamic_factors/_df_kernel — Python bindings.
//
// Exposes:
//   * fit_pca(returns, k_factors, shrinkage) → dict
//   * align_signs(curr, prev) → in-place flip of signs
//   * KalmanFilter3 / KalmanFilter5 — templated filter for k=3 and 5
//   * RollingStats — fixed-window mean / variance / z-score
//
// Numpy <-> Eigen conversion is handled via pybind11/eigen.h, which
// uses zero-copy views when input shapes/strides match.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "df/kalman.hpp"
#include "df/linalg.hpp"
#include "df/pca.hpp"
#include "df/rolling_stats.hpp"

namespace py = pybind11;
using namespace df;

// Templated helper: bind a KalmanFilter<K> with a given Python class
// name. Since K is a template parameter, we duplicate-instantiate.
template <int K>
void bind_kalman(py::module_& m, const char* name) {
    using KF = KalmanFilter<K>;
    py::class_<KF>(m, name)
        .def(py::init<const typename KF::State&,
                      const typename KF::StateCov&,
                      const typename KF::StateCov&,
                      double>(),
             py::arg("initial_state"), py::arg("initial_cov"),
             py::arg("process_noise"), py::arg("observation_noise"),
             "Kalman filter with K-dim state and scalar observations.")
        .def("step", &KF::step,
             py::arg("H"), py::arg("observed"),
             "Run one predict-update cycle. Returns the innovation.")
        .def_property_readonly("state",      &KF::state)
        .def_property_readonly("covariance", &KF::covariance)
        .def_property_readonly("log_likelihood", &KF::log_likelihood)
        .def("reset", &KF::reset,
             py::arg("initial_state"), py::arg("initial_cov"));
}

PYBIND11_MODULE(_df_kernel, m) {
    m.doc() = "dynamic-factors C++ kernel: PCA + Kalman + rolling stats.";

    // ---- PCA ----
    m.def("fit_pca",
          [](MatrixCRef returns, std::size_t k_factors, bool shrinkage) {
              const auto pca = fit_pca(returns, k_factors, shrinkage);
              py::dict out;
              out["eigenvalues"]    = pca.eigenvalues;
              out["loadings"]       = pca.loadings;
              out["factor_returns"] = pca.factor_returns;
              out["total_variance"] = pca.total_variance;
              return out;
          },
          py::arg("returns"), py::arg("k_factors"),
          py::arg("shrinkage") = true,
          "Fit PCA on a (T × N) returns matrix. Returns a dict with "
          "eigenvalues / loadings / factor_returns / total_variance.");

    // ---- Linalg primitives (exposed for Python tests) ----
    m.def("sample_covariance", &sample_covariance,
          py::arg("returns"));
    m.def("ledoit_wolf",       &ledoit_wolf,
          py::arg("returns"));
    m.def("symmetric_eigen",
          [](MatrixCRef A) {
              const auto e = symmetric_eigen(A);
              py::dict out;
              out["eigenvalues"]  = e.eigenvalues;
              out["eigenvectors"] = e.eigenvectors;
              return out;
          },
          py::arg("A"));

    // ---- Kalman filters ----
    bind_kalman<3>(m, "KalmanFilter3");
    bind_kalman<5>(m, "KalmanFilter5");

    // ---- RollingStats ----
    py::class_<RollingStats>(m, "RollingStats")
        .def(py::init<std::size_t>(), py::arg("window"))
        .def("push", &RollingStats::push, py::arg("x"),
             "Add a sample, return its rolling z-score.")
        .def_property_readonly("size", &RollingStats::size)
        .def_property_readonly("window", &RollingStats::window)
        .def_property_readonly("mean", &RollingStats::mean)
        .def_property_readonly("variance", &RollingStats::variance)
        .def("reset", &RollingStats::reset);
}
